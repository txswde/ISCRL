import os
import torch
import h5py
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
from models import DSRRL
from visualization.grad_cam import SmoothGradCAMpp
import argparse

# Configuration
FRAMES_ROOT = r"D:\papers\videos_15\summe_frames"

def load_model(checkpoint_path, input_dim=1024, hidden_dim=512, num_layers=2, rnn_cell='gru'):
    """Load the trained DSRRL model."""
    print(f"Loading model from {checkpoint_path}...")
    
    # Initialize model
    model = DSRRL(in_dim=input_dim, hid_dim=hidden_dim, num_layers=num_layers, cell=rnn_cell)
    
    # Load weights
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Handle DataParallel wrapper and state_dict extraction
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        print("Error: Unknown checkpoint format.")
        return None

    # Remove 'module.' prefix if present (from DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    model.eval()
    return model

def get_video_data(h5_path, key):
    with h5py.File(h5_path, 'r') as f:
        features = f[key]['features'][...]
        # Try to get video name
        try:
            video_name = f[key]['video_name'][()].decode('utf-8')
        except:
            video_name = None
            print("Warning: Could not find 'video_name' in dataset.")
            
    return features, video_name

def visualize_temporal(att_weights, output_path):
    """Plot temporal attention weights."""
    att_weights = att_weights.squeeze().cpu().numpy()
    
    # If 2D (matrix), take the mean attention received by each frame (column mean)
    # This represents "how much other frames attended to this frame"
    if len(att_weights.shape) == 2:
        att_weights = np.mean(att_weights, axis=0)

    plt.figure(figsize=(12, 4))
    plt.plot(att_weights, label='Avg Self-Attention Received', color='blue')
    plt.fill_between(range(len(att_weights)), att_weights, color='blue', alpha=0.1)
    
    plt.xlabel('Frame Index')
    plt.ylabel('Attention Weight')
    plt.title('Temporal Explainability: Attention over Time (DSRRL)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved temporal visualization to {output_path}")
    return att_weights

def visualize_spatial(video_name, video_key, important_indices, output_dir, top_k=5):
    """Generate Smooth Grad-CAM++ for top-k important frames."""
    print(f"Generating spatial visualizations for video: {video_name} (Key: {video_key})")
    
    grad_cam = SmoothGradCAMpp()
    
    # Try finding directory by name first, then by key
    video_dir = os.path.join(FRAMES_ROOT, video_name)
    if not os.path.exists(video_dir):
        # Try key
        video_dir_key = os.path.join(FRAMES_ROOT, video_key)
        if os.path.exists(video_dir_key):
            video_dir = video_dir_key
        else:
            print(f"Error: Frame directory not found via name '{video_name}' or key '{video_key}' in {FRAMES_ROOT}")
            return

    # Sort indices by importance (descending) but keep only top K
    # important_indices are already top-k, but let's iterate them
    
    for rank, idx in enumerate(important_indices):
        # Construct image filename
        # Based on file listing: frame_0000.jpg, frame_0015.jpg... (Step 15, 0-based)
        # Mapping: Feature Index i -> frame_{i*15:04d}.jpg
        frame_number = idx * 15
        img_name = f"frame_{frame_number:04d}.jpg"
        img_path = os.path.join(video_dir, img_name)
        
        if not os.path.exists(img_path):
             # Try fallback: maybe direct index?
            img_name_alt = f"img_{idx+1:05d}.jpg" 
            img_path_alt = os.path.join(video_dir, img_name_alt)
            if os.path.exists(img_path_alt):
                img_path = img_path_alt
            else:
                 # Try 0-based img_
                img_name_alt2 = f"img_{idx:05d}.jpg"
                if os.path.exists(os.path.join(video_dir, img_name_alt2)):
                     img_path = os.path.join(video_dir, img_name_alt2)
                else: 
                    print(f"Skipping feature idx {idx} (expected {img_name}), file not found.")
                    continue

        heatmap_img, _ = grad_cam.forward(img_path)
        
        if heatmap_img is not None:
            save_path = os.path.join(output_dir, f"spatial_top{rank+1}_frame{idx}.jpg")
            cv2.imwrite(save_path, heatmap_img)
            print(f"Saved spatial visualization to {save_path}")

def run_pipeline(checkpoint, dataset_path, video_key, output_root='visualization_results'):
    # 1. Load Model
    model = load_model(checkpoint)
    if model is None: return

    # 2. Get Data
    features, video_name = get_video_data(dataset_path, video_key)
    print(f"Processing Video Key: {video_key}, Name: {video_name}")
    
    if video_name is None:
        print("Cannot proceed with spatial visualization without video name.")
        return

    # Create output directory
    video_output_dir = os.path.join(output_root, video_name)
    os.makedirs(video_output_dir, exist_ok=True)

    # 3. Temporal Forward Pass
    seq = torch.from_numpy(features).unsqueeze(0) # (1, Seq, Dim)
    if torch.cuda.is_available():
        seq = seq.cuda()
        
    with torch.no_grad():
        # returns: p, out_lay, att_score, features_inv, att_weights_
        _, _, _, _, att_weights = model(seq)

    # 4. Visualize Temporal
    temporal_plot_path = os.path.join(video_output_dir, 'temporal_attention.png')
    att_weights_np = visualize_temporal(att_weights, temporal_plot_path)

    # 5. Select Top-K Frames
    top_k = 5
    # Get indices of top_k attention scores
    # att_weights_np shape is (SeqLen, 1) or (SeqLen,)
    # If using Multi-Head, might be (1, Heads, SeqLen, SeqLen).
    # models.py SelfAttention:
    # logits = torch.matmul(Q, K.transpose(1,0)) -> (N, N)
    # att_weights_ = softmax(logits) -> (N, N)
    # Then y = V * weights
    # WAIT. models.py:48: att_weights_ = nn.functional.softmax(logits, dim=-1)
    # It returns an (N, N) matrix! The full attention map!
    # Not just a per-frame score.
    
    # "Temporal Attention" usually means "Importance of each frame". 
    # But Self-Attention is pairwise.
    # How does DSRRL use this?
    # models.py:84: out_lay = att_score + h
    # models.py:85: p = torch.sigmoid(self.fc(out_lay))
    # `p` is the frame importance score!
    # `att_weights` is the internal mechanism.
    # If the user asks for "Displaying Temporal Attention Weights" (Self-Attention 权重), 
    # they might want the Attention Matrix (Heatmap N x N) OR the final importance scores (`p`).
    
    # Interpretation:
    # 1. `p` (output probability) = Final Keyframe Selection Score.
    # 2. `att_weights_` (N x N) = How frames relate to each other.
    
    # The user request says: "利用 Self-Attention 权重展示时序关注点"
    # Usually this means showing the Attention Matrix or the diagonal?
    # Or maybe the "attention distribution" for a specific query?
    # But here it's Self-Attention.
    
    # Let's visualize:
    # A. The Final Importance Score Curve `p` (This is the most "temporal attention" like thing for summarization).
    # B. The Self-Attention Matrix (as a heatmap) to show relationships.
    
    # Let's save both.
    
    # For Spatial selection, we should use the FRAMES with highest `p` (Probability of being a summary).
    # Because `p` is what the model "chose".
    
    # Redo step 3: Get `p` as well.
    with torch.no_grad():
        probs, _, _, _, att_matrix = model(seq)
        
    probs = probs.squeeze().cpu().numpy()
    att_matrix = att_matrix.squeeze().cpu().numpy() # (N, N)
    
    # 4. Unload Model to save GPU memory for Grad-CAM
    del model
    del seq
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Unloaded DSRRL model and cleared CUDA cache.")

    # 5. Visualize Temporal
    plt.figure(figsize=(12, 4))
    plt.plot(probs, label='Importance Score (P)', color='green')
    plt.xlabel('Frame Index')
    plt.ylabel('Probability')
    plt.title('Frame Importance Scores (Model Output)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(video_output_dir, 'importance_scores.png'))
    plt.close()

    # Visualize Attention Matrix
    plt.figure(figsize=(10, 10))
    plt.imshow(att_matrix, cmap='hot', interpolation='nearest')
    plt.title('Self-Attention Matrix')
    plt.xlabel('Key Frame')
    plt.ylabel('Query Frame')
    plt.colorbar()
    plt.savefig(os.path.join(video_output_dir, 'attention_matrix.png'))
    plt.close()
    
    print(f"Saved temporal visualizations to {video_output_dir}")

    # 6. Spatial on Top-K Importance Frames
    top_indices = np.argsort(probs)[::-1][:top_k]
    visualize_spatial(video_name, video_key, top_indices, video_output_dir, top_k=top_k)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to .pth.tar model')
    parser.add_argument('--dataset', type=str, default='datasets/eccv16_dataset_summe_google_pool5.h5')
    parser.add_argument('--key', type=str, default='video_1')
    args = parser.parse_args()
    
    run_pipeline(args.checkpoint, args.dataset, args.key)
