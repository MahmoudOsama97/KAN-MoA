import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import logging
from einops import rearrange
from .kan import KANLinear

# Setup logger for debugging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# ======================================================
# 0. Helpers & Basic Layers
# ======================================================

def _parse_list(arg_str, num_experts, dtype=int):
    """
    Parses a comma-separated string into a list of length num_experts.
    Cycles the list if it is shorter than num_experts.
    """
    if isinstance(arg_str, (int, float)):
        return [dtype(arg_str)] * num_experts
        
    # Handle string input, strip whitespace
    items = [dtype(x.strip()) for x in str(arg_str).split(',')]
    
    if len(items) < num_experts:
        # Cycle repeat
        items = (items * (num_experts // len(items) + 1))[:num_experts]
    return items

class KroneckerLinear(nn.Module):
    """Efficient Kronecker Product Linear Layer (W = A kron B)"""
    def __init__(self, in_features, out_features, n=4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n = int(n)
        
        if in_features % n != 0 or out_features % n != 0:
            # Fallback for dimensions not divisible by n
            self.use_dense = True
            self.dense = nn.Linear(in_features, out_features, bias=False)
        else:
            self.use_dense = False
            # A: n x n
            self.phm_rule = nn.Parameter(torch.Tensor(n, n))
            # B: out/n x in/n
            self.phm_weight = nn.Parameter(torch.Tensor(out_features // n, in_features // n))
            self.reset_parameters()

    def reset_parameters(self):
        if not self.use_dense:
            nn.init.normal_(self.phm_rule, std=0.01)
            nn.init.xavier_uniform_(self.phm_weight)

    def forward(self, x):
        if self.use_dense:
            return self.dense(x)
        # Construct Weight Matrix W = A (kron) B
        W = torch.kron(self.phm_rule, self.phm_weight)
        return F.linear(x, W)

class KANRouter(nn.Module):
    def __init__(self, in_dim, num_experts, grid_size=3):
        super().__init__()
        self.layer = KANLinear(in_dim, num_experts, grid_size=grid_size)
    
    def forward(self, x):
        return self.layer(x)

class SpectralRouter(nn.Module):
    """
    Routes tokens based on their power spectral density (frequency content).
    Idea 2: Frequency-Aware Routing.
    """
    def __init__(self, in_features, num_experts):
        super().__init__()
        # RFFT of a vector of size N returns N//2 + 1 complex numbers
        self.spectrum_dim = in_features // 2 + 1
        
        # A simple linear gate that looks at the frequency spectrum
        self.gate = nn.Linear(self.spectrum_dim, num_experts, bias=False)
        
        # LayerNorm on the spectrum to stabilize training (focus on relative distribution)
        self.spec_norm = nn.LayerNorm(self.spectrum_dim)

    def forward(self, x):
        try:
            # x shape: (Batch * Seq, Dim) or (Batch, Seq, Dim)
            if x.dim() == 3:
                x = rearrange(x, 'b s d -> (b s) d')
                
            # 1. Apply Real Fast Fourier Transform (RFFT) along the embedding dimension
            x_fft = torch.fft.rfft(x, dim=-1, norm='ortho')
            
            # 2. Compute Power Spectrum (Magnitude)
            spectrum = x_fft.abs() # Shape: (N, Dim//2 + 1)
            
            # 3. Normalize spectrum
            spectrum = self.spec_norm(spectrum)
            
            # 4. Compute Logits
            logits = self.gate(spectrum)
            
            return logits
        except Exception as e:
            logger.error(f"SpectralRouter Failed. Input shape: {x.shape}", exc_info=True)
            raise e

class CosineTopKGate(nn.Module):
    """Simple cosine-similarity based router with top-k selection"""
    def __init__(self, in_features, num_experts, k=1):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        # Learnable expert embeddings
        self.expert_embeddings = nn.Parameter(torch.randn(num_experts, in_features))
        nn.init.orthogonal_(self.expert_embeddings)
    
    def forward(self, x):
        # x: (batch, in_features)
        # Normalize
        x_norm = F.normalize(x, p=2, dim=-1)
        expert_norm = F.normalize(self.expert_embeddings, p=2, dim=-1)
        
        # Cosine similarity
        logits = torch.matmul(x_norm, expert_norm.t())  # (batch, num_experts)
        return logits

# ======================================================
# 1. MOE Base Class (Shared Logic)
# ======================================================

class MOEBase(nn.Module):
    """Base class for all Mixture of Experts modules with enhanced functionality"""
    
    def __init__(self):
        super().__init__()
        self.last_routing_weights = None
        self.last_expert_outputs = None
        self.training_step = 0
        self.enable_load_balance = True
        self.enable_grid_entropy = True
        self.enable_pairwise_ortho = True
        self.kan_grad_scale = 0.1
        
    def compute_load_balance_loss(self):
        """Compute load balancing loss - FIXED VERSION.
        Uses softer penalty that allows some imbalance for specialization."""
        if not hasattr(self, 'last_routing_weights') or self.last_routing_weights is None:
            return torch.tensor(0.0)
        
        routing_weights = self.last_routing_weights  # [batch, num_experts]
        if routing_weights.dim() != 2:
            return torch.tensor(0.0)
        
        # Average routing weights across batch
        avg_weights = routing_weights.mean(dim=0)  # [num_experts]
        
        # Target is uniform distribution
        num_experts = avg_weights.size(0)
        target = 1.0 / num_experts
        
        # Soft penalty: only penalize if any expert gets less than 50% of fair share
        # or more than 200% of fair share
        min_threshold = target * 0.5
        max_threshold = target * 2.0
        
        # Penalize under-utilization
        under_penalty = F.relu(min_threshold - avg_weights).sum()
        
        # Penalize over-utilization  
        over_penalty = F.relu(avg_weights - max_threshold).sum()
        
        loss = under_penalty + over_penalty
        
        return loss

    def compute_lipschitz_loss(self):
        """
        Lipschitz regularization for KAN layers.
        Encourages smoother spline functions by penalizing large weight norms.
        """
        try:
            device = None
            # Find device from first parameter
            for p in self.parameters():
                device = p.device
                break
            if device is None:
                device = 'cpu'
            
            lip_loss = torch.tensor(0.0, device=device)
            count = 0
            
            for m in self.modules():
                if isinstance(m, KANLinear):
                    # Penalize spline weight magnitude for smoother functions
                    if hasattr(m, 'spline_weight') and m.spline_weight is not None:
                        lip_loss = lip_loss + m.spline_weight.pow(2).mean()
                        count += 1
                    if hasattr(m, 'base_weight') and m.base_weight is not None:
                        lip_loss = lip_loss + m.base_weight.pow(2).mean() * 0.1
                        count += 1
            
            if count > 0:
                lip_loss = lip_loss / count
            
            return lip_loss
        except Exception as e:
            logger.error(f"Error in compute_lipschitz_loss: {e}")
            return torch.tensor(0.0)

    def compute_grid_entropy_loss(self):
        """Compute grid entropy loss - FIXED VERSION.
        Uses lower target entropy to allow more focused routing."""
        if not hasattr(self, 'last_routing_weights') or self.last_routing_weights is None:
            return torch.tensor(0.0)
        
        routing_weights = self.last_routing_weights  # [batch, num_experts]
        if routing_weights.dim() != 2:
            return torch.tensor(0.0)
        
        # Compute entropy of routing distribution
        # Add small epsilon for numerical stability
        eps = 1e-8
        entropy = -(routing_weights * torch.log(routing_weights + eps)).sum(dim=-1).mean()
        
        # Target entropy: log(1.5) ≈ 0.4 instead of log(2) ≈ 0.69
        # This allows more focused routing while still encouraging some diversity
        num_experts = routing_weights.size(1)
        target_entropy = min(0.4, math.log(num_experts) * 0.5)
        
        # Penalize deviation from target
        loss = (entropy - target_entropy).abs()
        
        return loss
    
    def compute_pairwise_orthogonality_loss(self):
        """Compute pairwise orthogonality loss between experts - FIXED VERSION.
        Only penalizes high similarity (>0.3) to allow some beneficial correlation."""
        if not hasattr(self, 'last_expert_outputs') or self.last_expert_outputs is None:
            return torch.tensor(0.0)
        
        expert_outputs = self.last_expert_outputs  # [batch, num_experts, dim]
        if expert_outputs.dim() != 3 or expert_outputs.size(1) < 2:
            return torch.tensor(0.0)
        
        # Normalize expert outputs
        expert_outputs_norm = F.normalize(expert_outputs, p=2, dim=-1)
        
        # Compute pairwise cosine similarities
        # [batch, num_experts, num_experts]
        similarities = torch.bmm(expert_outputs_norm, expert_outputs_norm.transpose(1, 2))
        
        # Create mask for off-diagonal elements
        num_experts = similarities.size(1)
        mask = ~torch.eye(num_experts, dtype=torch.bool, device=similarities.device)
        
        # Get off-diagonal similarities
        off_diag = similarities[:, mask].view(similarities.size(0), -1)
        
        # Only penalize similarities above threshold (0.3) - allows some correlation
        threshold = 0.3
        excess = F.relu(off_diag.abs() - threshold)
        
        # Mean of excess similarities
        loss = excess.mean()
        
        return loss
    
    def compute_orthogonality_loss(self):
        """Legacy orthogonality loss - falls back to pairwise if enabled"""
        if self.enable_pairwise_ortho:
            return self.compute_pairwise_orthogonality_loss()
        
        # Original implementation as fallback
        if not hasattr(self, 'last_expert_outputs') or self.last_expert_outputs is None:
            return torch.tensor(0.0)
        
        try:
            expert_outs = self.last_expert_outputs
            
            # Handle different tensor shapes
            while expert_outs.dim() > 3:
                expert_outs = expert_outs.squeeze(1)
                
            num_experts = expert_outs.shape[1]
            device = expert_outs.device
            
            if num_experts < 2 or not hasattr(self, 'grid_list'):
                return torch.tensor(0.0, device=device)
            
            median_grid = np.median(self.grid_list)
            coarse_idx = [i for i, g in enumerate(self.grid_list) if g <= median_grid]
            fine_idx = [i for i, g in enumerate(self.grid_list) if g > median_grid]
            
            if len(coarse_idx) == 0 or len(fine_idx) == 0:
                return torch.tensor(0.0, device=device)
            
            coarse_outs = expert_outs[:, coarse_idx, :].mean(dim=1)
            fine_outs = expert_outs[:, fine_idx, :].mean(dim=1)
            
            cosine_sim = F.cosine_similarity(coarse_outs, fine_outs, dim=-1)
            return cosine_sim.abs().mean()
        except Exception as e:
            logger.error(f"Error in legacy compute_orthogonality_loss: {e}")
            return torch.tensor(0.0)
    
    def scale_kan_gradients(self):
        """Scale gradients for stable KAN training"""
        try:
            for m in self.modules():
                if isinstance(m, KANLinear):
                    for p in m.parameters():
                        if p.grad is not None:
                            p.grad.data *= self.kan_grad_scale
            
            # Gradually increase scale (warm-up)
            self.kan_grad_scale = min(1.0, self.kan_grad_scale + 0.0001)
        except Exception as e:
            logger.error(f"Error in scale_kan_gradients: {e}")
    
    def increment_step(self):
        """Increment training step counter"""
        self.training_step += 1

# ======================================================
# 2. MOE KAN-Kronecker (Best Performing)
# ======================================================

class MOEKANKroneckerInjectedLinear(MOEBase):
    def __init__(self, in_features, out_features, bias=False, 
                 num_experts=4, r_arg="4", grid_arg="5", router_type='cosine',
                 enable_load_balance=True, enable_grid_entropy=True, 
                 enable_pairwise_ortho=True, kan_init_scale=0.1):
        super().__init__()
        
        # Store configurations
        self.enable_load_balance = enable_load_balance
        self.enable_grid_entropy = enable_grid_entropy
        self.enable_pairwise_ortho = enable_pairwise_ortho
        self.kan_init_scale = kan_init_scale
        
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.linear.weight.requires_grad = False
        if bias: self.linear.bias.requires_grad = False

        # Parse and STORE lists (Fixes AttributeError)
        self.r_list = _parse_list(r_arg, num_experts, int)
        self.grid_list = _parse_list(grid_arg, num_experts, int)
        self.bottleneck_norms = nn.ModuleList([
            nn.InstanceNorm1d(self.r_list[i]) for i in range(num_experts)
        ])         
        # Stability Fix: Normalize inputs for KAN stability
        self.input_norm = nn.LayerNorm(in_features)

        # Router Selection
        if router_type == 'kan':
            self.router = KANRouter(in_features, num_experts, grid_size=3)
        elif router_type == 'spectral':
            self.router = SpectralRouter(in_features, num_experts)
        else:
            self.router = CosineTopKGate(in_features, num_experts, k=1)

        self.expert_downs = nn.ModuleList()
        self.expert_kans = nn.ModuleList()
        self.expert_ups = nn.ModuleList()
        self.kan_input_norms = nn.ModuleList()  # Normalize inputs to KAN grid range
        
        for i in range(num_experts):
            n = self.r_list[i]
            g = self.grid_list[i]
            
            self.expert_downs.append(KroneckerLinear(in_features, n, n=n))
            self.kan_input_norms.append(nn.LayerNorm(n))  # Normalize to ~N(0,1) before KAN
            self.expert_kans.append(KANLinear(n, n, grid_size=g))
            self.expert_ups.append(KroneckerLinear(n, out_features, n=n))

        self.num_experts = num_experts
        self.scale = 1.0
        self.dropout = nn.Dropout(0.1)
        self.last_expert_outputs = None

        # Initialize KAN experts with warm-start for stability
        for i, kan in enumerate(self.expert_kans):
            try:
                # Small initial scale for stable training
                if hasattr(kan, 'base_weight') and kan.base_weight is not None:
                    nn.init.zeros_(kan.base_weight)
                if hasattr(kan, 'spline_weight') and kan.spline_weight is not None:
                    nn.init.normal_(kan.spline_weight, std=self.kan_init_scale * 0.01)
                logger.debug(f"Initialized KAN expert {i} with scale {self.kan_init_scale}")
            except Exception as e:
                logger.error(f"Failed to initialize KAN expert {i}: {e}")

    def forward(self, x):
        try:
            base_out = self.linear(x)
            b, s, d = x.shape
            x_flat = rearrange(x, 'b s d -> (b s) d')
            
            # Router computation
            router_logits = self.router(x_flat)
            routing_weights = F.softmax(router_logits, dim=-1)
            self.last_routing_weights = routing_weights
            
            # Store expert outputs for orthogonality loss
            expert_outputs_list = []
            
            output = torch.zeros((b * s, base_out.shape[-1]), device=x.device, dtype=x.dtype)
            
            # Find max dimension for padding
            max_r = max(self.r_list);
            
            for i, expert in enumerate(self.expert_kans):
                # Kronecker product computation
                x_down = self.expert_downs[i](x)  # (b, s, r)
                x_down_flat = rearrange(x_down, 'b s r -> (b s) r')
                
                # Normalize inputs to KAN grid range [-1, 1]
                x_down_norm = self.kan_input_norms[i](x_down_flat)
                
                # KAN transformation
                kan_out = expert(x_down_norm)  # (b*s, r_i)
                
                # Pad to max_r for stacking (for orthogonality loss)
                if kan_out.shape[-1] < max_r:
                    padding = max_r - kan_out.shape[-1]
                    kan_out_padded = F.pad(kan_out, (0, padding), mode='constant', value=0)
                else:
                    kan_out_padded = kan_out
                expert_outputs_list.append(kan_out_padded)  # (b*s, max_r)
                
                # Upproject (use original unpadded output)
                final_out = self.expert_ups[i](kan_out.view(b, s, -1))  # (b, s, out)
                final_out_flat = rearrange(final_out, 'b s o -> (b s) o')
                
                # Weighted aggregation
                weight = routing_weights[:, i:i+1]
                output = output + weight * final_out_flat * self.kan_init_scale
            
            # Store for loss computation: (N, num_experts, max_r)
            if expert_outputs_list:
                self.last_expert_outputs = torch.stack(expert_outputs_list, dim=1)
            
            return base_out + output.view(b, s, -1)
            
        except Exception as e:
            logger.error(f"Error in MOEKANKroneckerInjectedLinear forward: {e}")
            # Fallback to base linear
            return self.linear(x)

    def get_complexity(self):
        total = 0
        for kan in self.expert_kans:
            total += kan.regularization_loss(1.0, 0.0)
        return total

# ======================================================
# 3. MOE KAN-LoRA
# ======================================================

class MOEKANLoraInjectedLinear(MOEBase):
    def __init__(self, in_features, out_features, bias=False, 
                 num_experts=4, r_arg="8", grid_arg="5", router_type='cosine'):
        super().__init__()
        
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.linear.weight.requires_grad = False
        if bias: self.linear.bias.requires_grad = False

        self.r_list = _parse_list(r_arg, num_experts, int)
        self.grid_list = _parse_list(grid_arg, num_experts, int)
        
        self.input_norm = nn.LayerNorm(in_features)

        if router_type == 'kan':
            self.router = KANRouter(in_features, num_experts, grid_size=3)
        elif router_type == 'spectral':
            self.router = SpectralRouter(in_features, num_experts)
        else:
            self.router = CosineTopKGate(in_features, num_experts, k=1)

        self.expert_downs = nn.ModuleList()
        self.expert_kans = nn.ModuleList()
        self.expert_ups = nn.ModuleList()
        self.kan_input_norms = nn.ModuleList()  # Normalize inputs to KAN grid range
        
        for i in range(num_experts):
            r = self.r_list[i]
            g = self.grid_list[i]
            
            self.expert_downs.append(nn.Linear(in_features, r, bias=False))
            self.kan_input_norms.append(nn.LayerNorm(r))  # Normalize to ~N(0,1) before KAN
            self.expert_kans.append(KANLinear(r, r, grid_size=g))
            self.expert_ups.append(nn.Linear(r, out_features, bias=False))
            
            nn.init.kaiming_uniform_(self.expert_downs[i].weight, a=math.sqrt(5))
            nn.init.zeros_(self.expert_ups[i].weight)

        self.num_experts = num_experts
        self.scale = 1.0
        self.dropout = nn.Dropout(0.1)
        self.last_expert_outputs = None

    def forward(self, x):
        try:
            base_out = self.linear(x)
            x_norm = self.input_norm(x)
            b, s, d = x.shape
            x_flat = rearrange(x_norm, 'b s d -> (b s) d')
            
            router_logits = self.router(x_flat)
            if self.training:
                router_logits = router_logits + torch.randn_like(router_logits) / self.num_experts
            routing_weights = F.softmax(router_logits, dim=-1)
            self.last_routing_weights = routing_weights
            
            # Find max rank for padding (to enable stacking)
            max_r = max(self.r_list)
            
            outputs_list = []
            expert_hidden_list = []
            for i in range(self.num_experts):
                h = self.expert_downs[i](x_norm)  # (b, s, r_i)
                h_flat = rearrange(h, 'b s r -> (b s) r')
                
                # Normalize inputs to KAN grid range
                h_flat_norm = self.kan_input_norms[i](h_flat)
                h_kan = self.expert_kans[i](h_flat_norm)  # (b*s, r_i)
                
                # Pad for orthogonality loss
                if h_kan.shape[-1] < max_r:
                    h_kan_padded = F.pad(h_kan, (0, max_r - h_kan.shape[-1]))
                else:
                    h_kan_padded = h_kan
                expert_hidden_list.append(h_kan_padded)
                
                h_kan_3d = h_kan.view(b, s, -1)
                h_out = self.dropout(h_kan_3d)
                h_out = self.expert_ups[i](h_out)
                outputs_list.append(rearrange(h_out, 'b s d -> (b s) d'))
            
            # Store expert outputs: (N, num_experts, max_r)
            self.last_expert_outputs = torch.stack(expert_hidden_list, dim=1)
                
            expert_outputs_stacked = torch.stack(outputs_list, dim=1)  # (N, num_experts, out_dim)
            
            routing_weights_unsqueezed = routing_weights.unsqueeze(-1)
            moe_out_flat = torch.sum(routing_weights_unsqueezed * expert_outputs_stacked, dim=1)
            
            return base_out + (rearrange(moe_out_flat, '(b s) d -> b s d', b=b) * self.scale)
        except Exception as e:
            logger.error(f"Forward Pass Failed in MOEKANLoRA: {e}", exc_info=True)
            return self.linear(x)

    def get_complexity(self):
        total = 0
        for kan in self.expert_kans:
            total += kan.regularization_loss(1.0, 0.0)
        return total

# ======================================================
# 4. MOE Pure KAN (KAN -> KAN)
# ======================================================

class MOEPureKANInjectedLinear(MOEBase):
    def __init__(self, in_features, out_features, bias=False, 
                 num_experts=4, r_arg="8", grid_arg="5", router_type='cosine'):
        super().__init__()
        
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.linear.weight.requires_grad = False
        if bias: self.linear.bias.requires_grad = False

        self.r_list = _parse_list(r_arg, num_experts, int)
        self.grid_list = _parse_list(grid_arg, num_experts, int)
        
        self.input_norm = nn.LayerNorm(in_features)

        if router_type == 'kan':
            self.router = KANRouter(in_features, num_experts, grid_size=3)
        elif router_type == 'spectral':
            self.router = SpectralRouter(in_features, num_experts)
        else:
            self.router = CosineTopKGate(in_features, num_experts, k=1)

        self.expert_downs = nn.ModuleList()
        self.expert_ups = nn.ModuleList()
        self.kan_input_norms = nn.ModuleList()  # Normalize inputs to KAN grid range
        
        for i in range(num_experts):
            r = self.r_list[i]
            g = self.grid_list[i]
            
            self.kan_input_norms.append(nn.LayerNorm(in_features))  # Normalize before down KAN
            # Down: KAN(in, r)
            self.expert_downs.append(KANLinear(in_features, r, grid_size=g))
            # Up: KAN(r, out)
            kan_up = KANLinear(r, out_features, grid_size=g)
            
            # CRITICAL FIX: Zero Init for Pure KAN Up-Projection
            with torch.no_grad():
                kan_up.base_weight.zero_()
                kan_up.spline_weight.zero_()
                if kan_up.enable_standalone_scale_spline:
                    kan_up.spline_scaler.zero_()
            
            self.expert_ups.append(kan_up)

        self.num_experts = num_experts
        self.scale = 1.0
        self.dropout = nn.Dropout(0.1)
        self.last_expert_outputs = None

    def forward(self, x):
        try:
            base_out = self.linear(x)
            x_norm = self.input_norm(x)
            b, s, d = x.shape
            x_flat = rearrange(x_norm, 'b s d -> (b s) d')
            
            router_logits = self.router(x_flat)
            if self.training:
                router_logits = router_logits + torch.randn_like(router_logits) / self.num_experts
            routing_weights = F.softmax(router_logits, dim=-1)
            self.last_routing_weights = routing_weights
            
            outputs_list = []
            expert_hidden_list = []
            
            # Find max rank for padding
            max_r = max(self.r_list)
            
            for i in range(self.num_experts):
                # Normalize before KAN
                h_norm = self.kan_input_norms[i](x_norm)
                h = self.expert_downs[i](h_norm)  # (b, s, r_i)
                h_flat = rearrange(h, 'b s r -> (b s) r')
                
                # Pad for orthogonality loss
                if h_flat.shape[-1] < max_r:
                    h_padded = F.pad(h_flat, (0, max_r - h_flat.shape[-1]))
                else:
                    h_padded = h_flat
                expert_hidden_list.append(h_padded)
                
                h = self.dropout(h)
                h = self.expert_ups[i](h)
                outputs_list.append(rearrange(h, 'b s d -> (b s) d'))
            
            # Store expert hidden states: (N, num_experts, max_r)
            self.last_expert_outputs = torch.stack(expert_hidden_list, dim=1)
            
            expert_outputs_stacked = torch.stack(outputs_list, dim=1)
            
            routing_weights_unsqueezed = routing_weights.unsqueeze(-1)
            moe_out_flat = torch.sum(routing_weights_unsqueezed * expert_outputs_stacked, dim=1)
            
            return base_out + (rearrange(moe_out_flat, '(b s) d -> b s d', b=b) * self.scale)
        except Exception as e:
            logger.error(f"Forward Pass Failed in MOEPureKAN: {e}", exc_info=True)
            return self.linear(x)

    def get_complexity(self):
        total = 0
        for i in range(self.num_experts):
            total += self.expert_downs[i].regularization_loss(1.0, 0.0)
            total += self.expert_ups[i].regularization_loss(1.0, 0.0)
        return total

# ======================================================
# Injection Logic
# ======================================================

def _find_modules(model, target_list):
    matches = []
    for name, module in model.named_modules():
        if any(t in name for t in target_list) and isinstance(module, nn.Linear):
            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            child_name = name.rsplit('.', 1)[1] if '.' in name else name
            parent = model.get_submodule(parent_name)
            matches.append((parent, child_name, module))
    return matches

def inject_kan_moe(model, args):
    targets = args.injection_targets.split(',')
    print(f"Injecting {args.adapter_type} into {targets} with Router: {args.router_type}")
    
    matches = _find_modules(model, targets)
    
    for parent, name, child in matches:
        in_features = child.in_features
        out_features = child.out_features
        bias = child.bias is not None
        
        if args.adapter_type == 'kan_lora':
            new_layer = MOEKANLoraInjectedLinear(
                in_features, out_features, bias,
                num_experts=args.num_experts,
                r_arg=args.expert_ranks,        
                grid_arg=args.kan_grids,       
                router_type=args.router_type
            )
        elif args.adapter_type == 'kan_kronecker':
            new_layer = MOEKANKroneckerInjectedLinear(
                in_features, out_features, bias,
                num_experts=args.num_experts,
                r_arg=args.expert_ranks,
                grid_arg=args.kan_grids,
                router_type=args.router_type
            )
        elif args.adapter_type == 'pure_kan':
            new_layer = MOEPureKANInjectedLinear(
                in_features, out_features, bias,
                num_experts=args.num_experts,
                r_arg=args.expert_ranks,
                grid_arg=args.kan_grids,
                router_type=args.router_type
            )
        else:
            raise ValueError(f"Unknown type: {args.adapter_type}")
            
        new_layer.linear.weight = child.weight
        if bias: new_layer.linear.bias = child.bias
        new_layer.to(child.weight.device)
        setattr(parent, name, new_layer)
        
    for n, p in model.named_parameters():
        if "expert" in n or "router" in n or "phm" in n or "base_weight" in n or "spline" in n: 
            p.requires_grad = True
        else: 
            p.requires_grad = False
            
    return model

