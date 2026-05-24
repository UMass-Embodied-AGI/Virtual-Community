import torch

def get_robot_policy(jit_ckpt_path):
	policy = torch.jit.load(jit_ckpt_path)
	device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
	policy.to(device=device)
	return policy