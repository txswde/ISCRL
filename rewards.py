import torch
import sys
import torch.nn.functional as F

def compute_reward(seq, seq_inv, actions, beta=0.1, ignore_far_sim=True, temp_dist_thre=20, use_gpu=False):
    """
    Compute Dual-Space Reward (ISCRL)
    
    Args:
        seq: Original features (1, seq_len, 1024)
        seq_inv: Invariant features (1, seq_len, 128) - SimCLR projected
        actions: binary action sequence
        beta: Balance parameter (default 0.1)
    """
    
    # helper for single space reward
    def _get_space_reward(_features, _pick_idxs, _num_picks):
        if _num_picks == 0:
            return torch.tensor(0.).to(_features.device)
        
        _features = _features.squeeze() # (N, dim)
        n = _features.size(0)

        # Diversity
        if _num_picks == 1:
            r_div = torch.tensor(0.).to(_features.device)
        else:
            normed = F.normalize(_features, p=2, dim=1)
            dissim_mat = 1. - torch.matmul(normed, normed.t())
            dissim_submat = dissim_mat[_pick_idxs,:][:,_pick_idxs]
            if ignore_far_sim:
                pick_mat = _pick_idxs.expand(_num_picks, _num_picks)
                temp_dist_mat = torch.abs(pick_mat - pick_mat.t())
                dissim_submat[temp_dist_mat > temp_dist_thre] = 1.
            r_div = dissim_submat.sum() / (_num_picks * (_num_picks - 1.))
            
        # Representative
        dist_mat = torch.pow(_features, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist_mat = dist_mat + dist_mat.t()
        dist_mat.addmm_(_features, _features.t(), beta=1, alpha=-2)
        dist_mat = dist_mat[:,_pick_idxs]
        dist_mat = dist_mat.min(1, keepdim=True)[0]
        # logic from paper: exp(-mean_min_dist)
        r_rep = torch.exp(-dist_mat.mean())
        
        return (r_div + r_rep) * 0.5

    _actions = actions.detach()
    pick_idxs = _actions.squeeze().nonzero().squeeze()
    num_picks = len(pick_idxs) if pick_idxs.ndimension() > 0 else 1
    
    if num_picks == 0:
        reward = torch.tensor(0.)
        if use_gpu: reward = reward.cuda()
        return reward

    # Compute for Original Space
    # Detach features to prevent gradient flow through reward calculation input
    R_orig = _get_space_reward(seq.detach(), pick_idxs, num_picks)
    
    # Compute for Invariant Space
    R_inv = _get_space_reward(seq_inv.detach(), pick_idxs, num_picks)
    
    # Dual-Space Fusion
    reward = beta * R_orig + (1.0 - beta) * R_inv
    
    return reward
