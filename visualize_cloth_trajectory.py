#!/usr/bin/env python
"""
Visualize cloth manipulation from saved trajectory data
Creates beautiful 3D plots and videos for diploma presentation
"""

import sys
import pathlib
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless mode
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import imageio


def create_cloth_frame(ax, cloth_pos, goal_pos, title=""):
    """
    Create single frame showing cloth and goal
    
    Args:
        ax: matplotlib 3D axis
        cloth_pos: (9, 3) array of keypoint positions
        goal_pos: (9, 3) array of goal positions
        title: Frame title
    """
    ax.clear()
    
    # Plot cloth keypoints
    ax.scatter(cloth_pos[:, 0], cloth_pos[:, 1], cloth_pos[:, 2],
               c='blue', s=100, marker='o', label='Cloth', alpha=0.8)
    
    # Plot goal positions
    ax.scatter(goal_pos[:, 0], goal_pos[:, 1], goal_pos[:, 2],
               c='red', s=100, marker='x', label='Goal', alpha=0.6, linewidths=3)
    
    # Draw lines connecting cloth points (3x3 grid)
    cloth_grid = cloth_pos.reshape(3, 3, 3)
    
    # Horizontal lines
    for i in range(3):
        ax.plot(cloth_grid[i, :, 0], cloth_grid[i, :, 1], cloth_grid[i, :, 2],
                'b-', alpha=0.5, linewidth=1)
    
    # Vertical lines
    for j in range(3):
        ax.plot(cloth_grid[:, j, 0], cloth_grid[:, j, 1], cloth_grid[:, j, 2],
                'b-', alpha=0.5, linewidth=1)
    
    # Set labels and limits
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Set reasonable limits
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([0.5, 1.8])
    ax.set_zlim([0, 1.2])
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)


def visualize_episode(data_path: str, output_dir: str = None):
    """
    Create visualizations from trajectory data
    
    Args:
        data_path: Path to .pkl file with trajectory data
        output_dir: Output directory for images/videos
    """
    
    # Load data
    print(f"📂 Loading: {data_path}")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    episode_id = data['episode_id']
    trajectories = data['trajectories']
    goal_coords = data['goal_coords']
    logged_loss = data['logged_loss']
    replayed_loss = data['replayed_final_loss']
    actions_info = data['actions_info']
    
    print(f"\n📊 Episode: {episode_id}")
    print(f"   Actions: {len(trajectories)}")
    print(f"   Logged loss: {logged_loss:.6f} ({logged_loss * 100:.2f}cm)")
    print(f"   Replayed loss: {replayed_loss:.6f} ({replayed_loss * 100:.2f}cm)")
    
    # Setup output directory
    if output_dir is None:
        output_dir = pathlib.Path(data_path).parent / f"visualizations_{episode_id}"
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\n📁 Output directory: {output_dir}")
    
    # Create individual action visualizations
    print(f"\n🎨 Creating visualizations...")
    
    all_frames = []
    
    for action_idx, (traj, action_info) in enumerate(zip(trajectories, actions_info)):
        print(f"  Action {action_idx + 1}/{len(trajectories)}")
        
        # Get final position for this action
        final_pos = traj[-1]  # Shape: (9, 3)
        loss = action_info['loss']
        
        # Create figure
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        title = f"Action {action_idx + 1}/{len(trajectories)} | Loss: {loss * 100:.1f}cm"
        create_cloth_frame(ax, final_pos, goal_coords, title)
        
        # Save static image
        img_path = output_dir / f"action_{action_idx + 1:02d}.png"
        plt.savefig(img_path, dpi=150, bbox_inches='tight')
        
        # Save frame for video
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        all_frames.append(frame)
        
        plt.close(fig)
    
    print(f"✅ Created {len(all_frames)} static images")
    
    # Create video
    print(f"\n🎬 Creating video...")
    video_path = output_dir / f"cloth_manipulation_{episode_id}.mp4"
    
    # Duplicate each frame to make video slower
    video_frames = []
    for frame in all_frames:
        for _ in range(10):  # Show each action for 10 frames
            video_frames.append(frame)
    
    imageio.mimsave(str(video_path), video_frames, fps=10, quality=8)
    print(f"✅ Video saved: {video_path}")
    print(f"   Duration: {len(video_frames) / 10:.1f}s")
    
    # Create summary figure
    print(f"\n📈 Creating summary figure...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), subplot_kw={'projection': '3d'})
    axes = axes.flatten()
    
    # Show key actions
    key_indices = [0, 3, 7, 11, 14, 15]  # Beginning, middle steps, end
    for idx, action_idx in enumerate(key_indices):
        if action_idx < len(trajectories):
            traj = trajectories[action_idx]
            final_pos = traj[-1]
            loss = actions_info[action_idx]['loss']
            
            title = f"Step {action_idx + 1} | {loss * 100:.1f}cm"
            create_cloth_frame(axes[idx], final_pos, goal_coords, title)
    
    plt.suptitle(f"Cloth Manipulation Progress - {episode_id}\n"
                 f"Logged: {logged_loss * 100:.2f}cm | Replayed: {replayed_loss * 100:.2f}cm",
                 fontsize=16, fontweight='bold')
    
    summary_path = output_dir / f"summary_{episode_id}.png"
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✅ Summary saved: {summary_path}")
    
    # Create loss plot
    print(f"\n📉 Creating loss plot...")
    losses = [info['loss'] * 100 for info in actions_info]  # Convert to cm
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(1, len(losses) + 1), losses, 'b-o', linewidth=2, markersize=8)
    ax.axhline(y=logged_loss * 100, color='r', linestyle='--', label=f'Logged: {logged_loss * 100:.2f}cm')
    ax.axhline(y=replayed_loss * 100, color='g', linestyle='--', label=f'Replayed: {replayed_loss * 100:.2f}cm')
    
    ax.set_xlabel('Action Step', fontsize=14)
    ax.set_ylabel('Loss (cm)', fontsize=14)
    ax.set_title(f'Cloth Manipulation Loss Over Time - {episode_id}', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    loss_plot_path = output_dir / f"loss_plot_{episode_id}.png"
    plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✅ Loss plot saved: {loss_plot_path}")
    
    print(f"\n✅ All visualizations created!")
    print(f"\n📋 Files created:")
    print(f"   - {len(all_frames)} individual action images")
    print(f"   - 1 video: {video_path.name}")
    print(f"   - 1 summary: {summary_path.name}")
    print(f"   - 1 loss plot: {loss_plot_path.name}")
    
    return output_dir


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_cloth_trajectory.py <trajectory_data.pkl> [output_dir]")
        sys.exit(1)
    
    data_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        visualize_episode(data_path, output_dir)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
