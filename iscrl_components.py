import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SimCLRProjector(nn.Module):
    """
    Projection Head for SimCLR: Projects features to an invariant space.
    Structure: MLP (Linear -> ReLU -> Linear)
    """
    def __init__(self, input_dim=1024, hidden_dim=512, output_dim=128):
        super(SimCLRProjector, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class SimCLRLoss(nn.Module):
    """
    SimCLR Loss (NT-Xent) with feature-level augmentation.
    Since we work with pre-extracted features, we simulate augmentation via noise and masking.
    """
    def __init__(self, temperature=0.5, noise_std=0.05, mask_prob=0.2):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature
        self.noise_std = noise_std
        self.mask_prob = mask_prob

    def augment(self, x):
        """
        Feature-level augmentation:
        1. Add Gaussian noise
        2. Randomly mask (dropout) some elements
        """
        # Add noise
        noise = torch.randn_like(x) * self.noise_std
        augmented = x + noise
        
        # Apply random mask (dropout)
        mask = (torch.rand_like(x) > self.mask_prob).float()
        return augmented * mask

    def forward(self, features):
        """
        Args:
            features: (N, dim) Projected invariant features.
        """
        batch_size = features.shape[0]
        
        # Generate two views
        z_i = self.augment(features)
        z_j = self.augment(features)
        
        # Normalize
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Concatenate
        representations = torch.cat([z_i, z_j], dim=0)
        
        # Similarity matrix
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        # Mask out self-similarity
        mask = torch.eye(2 * batch_size).to(features.device).bool()
        similarity_matrix.masked_fill_(mask, -9e15)
        
        # Select positives (i and i+batch_size are pairs)
        pos_mask = torch.zeros((2 * batch_size, 2 * batch_size)).to(features.device).bool()
        for i in range(batch_size):
            pos_mask[i, i + batch_size] = True
            pos_mask[i + batch_size, i] = True
            
        positives = similarity_matrix[pos_mask].view(2 * batch_size, -1)
        negatives = similarity_matrix[~pos_mask].view(2 * batch_size, -1)
        
        logits = torch.cat([positives, negatives], dim=1)
        # Positive is always at index 0
        labels = torch.zeros(2 * batch_size).to(features.device).long()
        
        logits /= self.temperature
        
        return F.cross_entropy(logits, labels)

class AIMOptimizer:
    """
    AIM (Adaptive Intervention Mechanism)
    Dynamically adjusts Gradient Clipping and Learning Rate based on Reward Gain (G_t).
    """
    def __init__(self, I_min=0.1, I_max=2.0, lambda_reg=0.1, base_clip=5.0):
        self.I_t = 1.0 # Initial intervention intensity
        self.I_prev = 1.0
        self.I_min = I_min
        self.I_max = I_max
        self.lambda_reg = lambda_reg
        self.base_clip = base_clip
        
        self.baseline_reward_ema = None # EMA of reward
        self.prev_reward = None
        
    def update(self, current_reward):
        """
        Updates I_t based on G_t = Delta_R - lambda * |Delta_I|
        Returns: (clip_norm, lr_scale)
        """
        if self.prev_reward is None:
            self.prev_reward = current_reward
            self.last_Gt = 0.0
            return self.base_clip, 1.0
            
        # Delta R
        delta_R = current_reward - self.prev_reward
        
        # Delta I (change from last step)
        delta_I = self.I_t - self.I_prev
        
        # Gain function
        G_t = delta_R - self.lambda_reg * abs(delta_I)
        self.last_Gt = G_t
        
        # Update I_prev before changing I_t
        self.I_prev = self.I_t
        
        # Adjustment Logic
        if G_t >= 0:
            # Improvement: Increase intervention (tighten constraints)
            self.I_t = min(self.I_t * 1.1, self.I_max)
        else:
            # Degradation: Decrease intervention (loosen constraints)
            self.I_t = max(self.I_t * 0.95, self.I_min)
            
        self.prev_reward = current_reward
        
        # Calculate dynamic parameters
        # Higher I_t -> Lower clip_norm (Strict clipping)
        clip_norm = self.base_clip / self.I_t
        
        # Higher I_t -> Lower LR (Finer steps)
        # lr_scale = 1 / (1 + 0.1 * (I_t - 1))
        lr_scale = 1.0 / (1.0 + 0.1 * (self.I_t - 1.0))
        
        return clip_norm, lr_scale
