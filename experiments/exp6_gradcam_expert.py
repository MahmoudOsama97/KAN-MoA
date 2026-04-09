#!/usr/bin/env python3
"""
Experiment 6: GradCAM Conditioned on Expert
=============================================
Generates gradient-weighted attention maps for individual experts.
Shows what spatial regions each expert focuses on.

Hypothesis: Coarse experts (low G) attend to global structures/shapes
            Fine experts (high G) attend to textures/local patterns

Usage:
  python exp6_gradcam_expert.py \
      --checkpoint path/to/TE0.pth \
      --data_dir data \
      --dataset PACS \
      --output_dir experiments/exp6_results \
      --num_images 8
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns


def load_model_and_data(ckpt_path, data_dir, dataset_name):
    """Load model from checkpoint."""
    from domainbed import hparams_registry
    from domainbed.datasets import get_dataset
    from domainbed.lib.fast_data_loader import FastDataLoader
    from sconf import Config
    
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    args_dict = ckpt['args']
    hparams_dict = ckpt['model_hparams']
    test_envs = ckpt['test_envs']
    
    args = argparse.Namespace(**args_dict)
    args.data_dir = Path(data_dir)
    
    hparams = hparams_registry.default_hparams(args.algorithm, dataset_name)
    hparams.update(hparams_dict)
    hparams = Config(default=hparams)
    
    dataset, in_splits, out_splits = get_dataset(test_envs, args, hparams)
    
    from domainbed.algorithms import algorithms as algo_module
    algorithm_class = getattr(algo_module, args.algorithm)
    algorithm = algorithm_class(
        dataset.input_shape, dataset.num_classes, len(dataset) - len(test_envs), hparams, args
    )
    algorithm.load_state_dict(ckpt['model_dict'])
    algorithm.cuda()
    
    # Get test env loader
    eval_loaders = {}
    for i, env_name in enumerate(dataset.environments):
        ds = out_splits[i][0]
        loader = FastDataLoader(dataset=ds, batch_size=1, num_workers=2)
        eval_loaders[env_name] = loader
    
    return algorithm, dataset, eval_loaders, test_envs


class ExpertGradCAM:
    """
    Computes gradient-weighted attention maps conditioned on specific expert outputs.
    
    For ViT: Uses attention weights from intermediate layers as "feature maps"
    and computes gradient-weighted importance per expert.
    """
    
    def __init__(self, algorithm):
        from domainbed.models.kan_adapters import MOEBase
        self.algorithm = algorithm
        self.adapters = [(n, m) for n, m in algorithm.named_modules() if isinstance(m, MOEBase)]
        self.attention_maps = {}
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks on attention layers to capture attention maps."""
        for name, module in self.algorithm.named_modules():
            if 'attn' in name and hasattr(module, 'attn_drop'):
                # This is likely a ViT attention module
                hook = module.register_forward_hook(self._attn_hook_fn(name))
                self.hooks.append(hook)
    
    def _attn_hook_fn(self, name):
        def hook(module, input, output):
            # For timm ViT, attention output is the projected output
            # We store the input to get Q, K for attention computation
            if isinstance(input, tuple):
                self.attention_maps[name] = input[0].detach()
            else:
                self.attention_maps[name] = input.detach()
        return hook
    
    def compute_expert_gradcam(self, x, expert_idx, target_adapter_idx=None):
        """
        Compute GradCAM-like visualization for a specific expert.
        
        Strategy: 
        1. Forward pass to get expert output
        2. Backprop from expert's contribution to get gradients
        3. Use gradients to weight spatial attention
        """
        self.algorithm.eval()
        x = x.requires_grad_(True)
        
        # Forward pass
        logits = self.algorithm.predict(x)
        
        # Get the target adapter
        if target_adapter_idx is None:
            target_adapter_idx = len(self.adapters) // 2  # Middle layer
        
        if target_adapter_idx >= len(self.adapters):
            target_adapter_idx = 0
        
        adapter_name, adapter = self.adapters[target_adapter_idx]
        
        # Get expert-specific output
        if adapter.last_expert_outputs is None:
            print("Warning: No expert outputs captured")
            return None
        
        expert_outs = adapter.last_expert_outputs  # (N, num_experts, dim)
        
        # Target: maximize the specific expert's output norm
        expert_output = expert_outs[:, expert_idx, :]  # (N, dim)
        target = expert_output.norm(dim=-1).mean()
        
        # Backward
        self.algorithm.zero_grad()
        target.backward(retain_graph=True)
        
        # Get input gradients as saliency
        if x.grad is not None:
            grad = x.grad.detach()  # (1, C, H, W)
            
            # Average absolute gradients across channels
            saliency = grad.abs().mean(dim=1)  # (1, H, W)
            saliency = saliency.squeeze(0).cpu().numpy()
            
            # Normalize
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
            
            return saliency
        
        return None
    
    def compute_attention_based_cam(self, x, expert_idx, target_layer=-1):
        """
        Alternative: Compute attention-weighted CAM for ViT.
        Uses attention weights × expert routing to visualize per-expert focus.
        """
        self.algorithm.eval()
        self.attention_maps.clear()
        
        with torch.no_grad():
            logits = self.algorithm.predict(x)
        
        # Get routing weights for the specific expert
        adapter_name, adapter = self.adapters[min(abs(target_layer), len(self.adapters) - 1)]
        
        if adapter.last_routing_weights is None:
            return None
        
        routing_weights = adapter.last_routing_weights.detach().cpu()  # (N*seq, num_experts)
        expert_weight = routing_weights[:, expert_idx]  # (N*seq,)
        
        # For ViT: seq includes [CLS] + patches
        # Reshape to spatial (exclude CLS token)
        num_patches = expert_weight.shape[0] // x.shape[0]
        if num_patches > 1:
            expert_weight = expert_weight.view(x.shape[0], num_patches)
            
            # Exclude CLS token (first position) if present
            if num_patches == 197:  # 14*14 + 1 CLS for ViT-B/16 224px
                spatial_weight = expert_weight[:, 1:]  # (B, 196)
                h = w = 14
            else:
                spatial_weight = expert_weight
                h = w = int(np.sqrt(spatial_weight.shape[-1]))
                if h * w != spatial_weight.shape[-1]:
                    h = w = spatial_weight.shape[-1]  # fallback to 1D
            
            cam = spatial_weight.view(x.shape[0], h, w).numpy()
            
            # Normalize per image
            for b in range(cam.shape[0]):
                cam[b] = (cam[b] - cam[b].min()) / (cam[b].max() - cam[b].min() + 1e-8)
            
            return cam
        
        return None
    
    def cleanup(self):
        for hook in self.hooks:
            hook.remove()


def denormalize_image(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Convert normalized image tensor back to displayable numpy array."""
    img = tensor.cpu().clone()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    img = img.clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def main():
    parser = argparse.ArgumentParser(description="Exp6: GradCAM Conditioned on Expert")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--dataset", type=str, default="PACS")
    parser.add_argument("--output_dir", type=str, default="experiments/exp6_results")
    parser.add_argument("--num_images", type=int, default=8)
    parser.add_argument("--method", type=str, default="attention", 
                        choices=["gradient", "attention"],
                        help="CAM method: gradient-based or attention-based")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading checkpoint: {args.checkpoint}")
    algorithm, dataset, eval_loaders, test_envs = load_model_and_data(
        args.checkpoint, args.data_dir, args.dataset
    )
    
    env_names = dataset.environments
    
    from domainbed.models.kan_adapters import MOEBase
    adapters = [(n, m) for n, m in algorithm.named_modules() if isinstance(m, MOEBase)]
    grid_sizes = adapters[0][1].grid_list if adapters and hasattr(adapters[0][1], 'grid_list') else [3, 5, 10, 20]
    num_experts = len(grid_sizes)
    
    print(f"Grid sizes: {grid_sizes}")
    print(f"Environments: {env_names}")
    print(f"Method: {args.method}")
    
    gradcam = ExpertGradCAM(algorithm)
    
    # ======================================================================
    # Generate visualizations for sample images from each domain
    # ======================================================================
    for env_name, loader in eval_loaders.items():
        print(f"\nProcessing domain: {env_name}")
        
        images_processed = 0
        all_cams = []  # List of (image, expert_cams_dict)
        
        for batch in loader:
            if images_processed >= args.num_images:
                break
            
            x = batch["x"].cuda()
            y = batch["y"].item()
            
            expert_cams = {}
            for expert_idx in range(num_experts):
                if args.method == "gradient":
                    cam = gradcam.compute_expert_gradcam(x, expert_idx)
                else:
                    cam = gradcam.compute_attention_based_cam(x, expert_idx)
                
                if cam is not None:
                    if cam.ndim == 3:
                        cam = cam[0]  # Take first in batch
                    expert_cams[expert_idx] = cam
            
            if expert_cams:
                img = denormalize_image(x[0])
                all_cams.append((img, expert_cams, y))
                images_processed += 1
        
        if not all_cams:
            print(f"  No valid CAMs generated for {env_name}")
            continue
        
        # Plot: rows=images, cols=original + experts
        num_imgs = len(all_cams)
        fig, axes = plt.subplots(num_imgs, num_experts + 1, 
                                  figsize=(3 * (num_experts + 1), 3 * num_imgs))
        if num_imgs == 1:
            axes = axes.reshape(1, -1)
        
        for img_idx, (img, expert_cams, label) in enumerate(all_cams):
            # Original image
            axes[img_idx, 0].imshow(img)
            axes[img_idx, 0].set_title(f"Class {label}", fontsize=9)
            axes[img_idx, 0].axis('off')
            
            for expert_idx in range(num_experts):
                ax = axes[img_idx, expert_idx + 1]
                
                if expert_idx in expert_cams:
                    cam = expert_cams[expert_idx]
                    
                    # Resize CAM to image size
                    if cam.shape[0] != img.shape[0] or cam.shape[1] != img.shape[1]:
                        cam_tensor = torch.tensor(cam).unsqueeze(0).unsqueeze(0).float()
                        cam_resized = F.interpolate(
                            cam_tensor, size=(img.shape[0], img.shape[1]), 
                            mode='bilinear', align_corners=False
                        )
                        cam = cam_resized.squeeze().numpy()
                    
                    # Overlay
                    ax.imshow(img)
                    ax.imshow(cam, cmap='jet', alpha=0.5, vmin=0, vmax=1)
                else:
                    ax.imshow(img)
                    ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes, 
                            ha='center', va='center', fontsize=12, color='red')
                
                if img_idx == 0:
                    ax.set_title(f"E{expert_idx} (G={grid_sizes[expert_idx]})", 
                                fontsize=10, fontweight='bold')
                ax.axis('off')
        
        plt.suptitle(f"Expert Attention Maps - {env_name} ({args.dataset})\n"
                     f"Coarse experts → global shapes | Fine experts → local textures",
                     fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 
                    f"exp6_{args.dataset}_{env_name}_gradcam.pdf"), dpi=200, bbox_inches='tight')
        plt.savefig(os.path.join(args.output_dir, 
                    f"exp6_{args.dataset}_{env_name}_gradcam.png"), dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved {num_imgs} images for {env_name}")
    
    # ======================================================================
    # Generate summary: average CAM per expert across all domains
    # ======================================================================
    print("\nGenerating average CAM comparison...")
    
    avg_cam_per_expert = defaultdict(list)
    
    for env_name, loader in eval_loaders.items():
        count = 0
        for batch in loader:
            if count >= 5:
                break
            x = batch["x"].cuda()
            
            for expert_idx in range(num_experts):
                if args.method == "attention":
                    cam = gradcam.compute_attention_based_cam(x, expert_idx)
                else:
                    cam = gradcam.compute_expert_gradcam(x, expert_idx)
                
                if cam is not None:
                    if cam.ndim == 3:
                        cam = cam[0]
                    avg_cam_per_expert[expert_idx].append(cam)
            count += 1
    
    # Compute and plot spatial statistics
    fig, axes = plt.subplots(1, num_experts, figsize=(5 * num_experts, 4))
    if num_experts == 1:
        axes = [axes]
    
    for expert_idx in range(num_experts):
        ax = axes[expert_idx]
        if avg_cam_per_expert[expert_idx]:
            # Average CAM
            cams = avg_cam_per_expert[expert_idx]
            # Resize all to same size
            target_h, target_w = 14, 14
            resized = []
            for c in cams:
                ct = torch.tensor(c).unsqueeze(0).unsqueeze(0).float()
                cr = F.interpolate(ct, size=(target_h, target_w), mode='bilinear', align_corners=False)
                resized.append(cr.squeeze().numpy())
            
            avg_cam = np.mean(resized, axis=0)
            im = ax.imshow(avg_cam, cmap='hot', interpolation='bilinear')
            plt.colorbar(im, ax=ax, fraction=0.046)
            
            # Compute spatial entropy
            flat = avg_cam.flatten()
            flat = flat / (flat.sum() + 1e-10)
            entropy = -(flat * np.log(flat + 1e-10)).sum()
            ax.set_xlabel(f"Spatial Entropy: {entropy:.2f}", fontsize=10)
        
        ax.set_title(f"Expert {expert_idx} (G={grid_sizes[expert_idx]})", fontsize=12)
        ax.axis('off')
    
    plt.suptitle(f"Average Expert Attention ({args.dataset})\n"
                 f"High entropy = diffuse (global) | Low entropy = focused (local)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"exp6_{args.dataset}_avg_cam.pdf"), dpi=300)
    plt.savefig(os.path.join(args.output_dir, f"exp6_{args.dataset}_avg_cam.png"), dpi=300)
    plt.close()
    
    gradcam.cleanup()
    
    print(f"\nResults saved to {args.output_dir}")
    print(f"Files generated:")
    for f in sorted(os.listdir(args.output_dir)):
        if args.dataset in f:
            print(f"  - {f}")


if __name__ == "__main__":
    main()
