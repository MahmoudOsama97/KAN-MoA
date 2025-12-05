import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
import numpy as np
from domainbed.models.kan import KANLinear
from domainbed.models.lora.lora_util import CosineTopKGate

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
            print(f"\n[ERROR] SpectralRouter Failed.")
            print(f"Input shape: {x.shape}")
            raise e

# ======================================================
# 1. MOE Base Class (Shared Logic)
# ======================================================

class MOEBase(nn.Module):
    """
    Base class for KAN MoE Adapters containing shared loss logic.
    """
    def compute_orthogonality_loss(self):
        """Idea 3: Penalize dot product between Coarse and Fine experts."""
        try:
            # Check if forward pass has run and data exists
            if not hasattr(self, 'last_expert_outputs') or self.last_expert_outputs is None: 
                return 0.0
            
            # Check if grid_list exists (fix for AttributeError)
            if not hasattr(self, 'grid_list'):
                return 0.0

            # Determine split based on median grid size
            median_grid = np.median(self.grid_list)
            
            # Identify indices
            coarse_idx = [i for i, g in enumerate(self.grid_list) if g <= median_grid]
            fine_idx = [i for i, g in enumerate(self.grid_list) if g > median_grid]
            
            # If we don't have both groups (e.g., all grids are same), skip loss
            if not coarse_idx or not fine_idx: 
                return 0.0

            # outputs: (Batch*Seq, NumExperts, OutDim)
            expert_outs = self.last_expert_outputs
            
            # Sum outputs per group (Batch, OutDim)
            coarse_sum = expert_outs[:, coarse_idx, :].sum(dim=1)
            fine_sum = expert_outs[:, fine_idx, :].sum(dim=1)
            
            # Normalize vectors (Cosine Similarity approach)
            # Add eps to prevent div by zero
            coarse_norm = F.normalize(coarse_sum, p=2, dim=1)
            fine_norm = F.normalize(fine_sum, p=2, dim=1)
            
            # Squared Dot Product: || u.T @ v ||^2
            ortho_loss = (coarse_norm * fine_norm).sum(dim=1).pow(2).mean()
            return ortho_loss
            
        except Exception as e:
            # Safe fail: print debug info only if it crashes
            print(f"\n[ERROR] Orthogonality Loss Failed.")
            if hasattr(self, 'grid_list'): print(f"Grid List: {self.grid_list}")
            if hasattr(self, 'last_expert_outputs') and self.last_expert_outputs is not None:
                print(f"Outputs shape: {self.last_expert_outputs.shape}")
            raise e

# ======================================================
# 2. MOE KAN-Kronecker (Best Performing)
# ======================================================

class MOEKANKroneckerInjectedLinear(MOEBase):
    def __init__(self, in_features, out_features, bias=False, 
                 num_experts=4, r_arg="4", grid_arg="5", router_type='cosine'):
        super().__init__()       
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
        
        for i in range(num_experts):
            n = self.r_list[i]
            g = self.grid_list[i]
            
            self.expert_downs.append(KroneckerLinear(in_features, n, n=n))
            self.expert_kans.append(KANLinear(n, n, grid_size=g))
            self.expert_ups.append(KroneckerLinear(n, out_features, n=n))

        self.num_experts = num_experts
        self.scale = 1.0
        self.dropout = nn.Dropout(0.1)
        self.last_expert_outputs = None

    def forward(self, x):
        try:
            base_out = self.linear(x)
            
            # Apply LayerNorm before experts/router
            x_norm = self.input_norm(x)
            
            b, s, d = x.shape
            x_flat = rearrange(x_norm, 'b s d -> (b s) d')
            
            # Router
            router_logits = self.router(x_flat)
            if self.training:
                router_logits = router_logits + torch.randn_like(router_logits) / self.num_experts
            routing_weights = F.softmax(router_logits, dim=-1)
            self.last_routing_weights = routing_weights
            
            expert_outputs = []
            for i in range(self.num_experts):
                # Down (Kron) -> KAN -> Drop -> Up (Kron)
                # h comes from linear: (B, S, r)
                h = self.expert_downs[i](x_norm) 

                # Transpose for InstanceNorm: (B, r, S)
                h = h.transpose(1, 2) 

                # Normalize
                h = self.bottleneck_norms[i](h) 

                # Transpose back for KAN: (B, S, r)
                # KANLinear in efficient-kan expects (..., Dim)
                h = h.transpose(1, 2) 

                h = self.expert_kans[i](h)
                h = self.dropout(h)
                h = self.expert_ups[i](h)        # (B, S, Out)
                expert_outputs.append(rearrange(h, 'b s d -> (b s) d'))
                
            # Stack for calculations: (N, Experts, Dim)
            expert_outputs_stacked = torch.stack(expert_outputs, dim=1)
            
            # Store for Orthogonality Loss (Idea 3)
            self.last_expert_outputs = expert_outputs_stacked
            
            # Weighted Sum
            routing_weights_unsqueezed = routing_weights.unsqueeze(-1)
            moe_out_flat = torch.sum(routing_weights_unsqueezed * expert_outputs_stacked, dim=1)
            
            return base_out + (rearrange(moe_out_flat, '(b s) d -> b s d', b=b) * self.scale)
            
        except Exception as e:
            print(f"\n[ERROR] Forward Pass Failed in MOEKANKronecker.")
            print(f"Input x shape: {x.shape}")
            print(f"Num Experts: {self.num_experts}")
            raise e

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
        
        for i in range(num_experts):
            r = self.r_list[i]
            g = self.grid_list[i]
            
            self.expert_downs.append(nn.Linear(in_features, r, bias=False))
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
            
            outputs_list = []
            for i in range(self.num_experts):
                h = self.expert_downs[i](x_norm)
                h = self.expert_kans[i](h)
                h = self.dropout(h)
                h = self.expert_ups[i](h)
                outputs_list.append(rearrange(h, 'b s d -> (b s) d'))
                
            expert_outputs_stacked = torch.stack(outputs_list, dim=1)
            self.last_expert_outputs = expert_outputs_stacked
            
            routing_weights_unsqueezed = routing_weights.unsqueeze(-1)
            moe_out_flat = torch.sum(routing_weights_unsqueezed * expert_outputs_stacked, dim=1)
            
            return base_out + (rearrange(moe_out_flat, '(b s) d -> b s d', b=b) * self.scale)
        except Exception as e:
            print(f"\n[ERROR] Forward Pass Failed in MOEKANLoRA.")
            raise e

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
        
        for i in range(num_experts):
            r = self.r_list[i]
            g = self.grid_list[i]
            
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
            for i in range(self.num_experts):
                h = self.expert_downs[i](x_norm)
                h = self.dropout(h)
                h = self.expert_ups[i](h)
                outputs_list.append(rearrange(h, 'b s d -> (b s) d'))
                
            expert_outputs_stacked = torch.stack(outputs_list, dim=1)
            self.last_expert_outputs = expert_outputs_stacked
            
            routing_weights_unsqueezed = routing_weights.unsqueeze(-1)
            moe_out_flat = torch.sum(routing_weights_unsqueezed * expert_outputs_stacked, dim=1)
            
            return base_out + (rearrange(moe_out_flat, '(b s) d -> b s d', b=b) * self.scale)
        except Exception as e:
            print(f"\n[ERROR] Forward Pass Failed in MOEPureKAN.")
            raise e

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

