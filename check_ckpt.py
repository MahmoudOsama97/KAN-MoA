import torch
import sys
path = sys.argv[1]
ckpt = torch.load(path, map_location='cpu', weights_only=False)
h = ckpt['model_hparams']
print('grids:', h.get('kan_grids'))
print('test_envs:', ckpt['test_envs'])
print('num_experts:', h.get('num_experts'))
print('expert_ranks:', h.get('expert_ranks'))
