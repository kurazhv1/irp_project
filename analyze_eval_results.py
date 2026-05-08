#!/usr/bin/env python3
"""
Analyze evaluation results and create visualizations for diploma thesis.
Processes action logs from eval_irp_cloth_sim.py evaluation runs.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def load_all_logs(log_dir):
    """Load all action logs from directory."""
    log_dir = Path(log_dir)
    logs = []
    
    for log_file in sorted(log_dir.glob("action_log_*.json")):
        with open(log_file, 'r') as f:
            data = json.load(f)
            logs.append(data)
    
    print(f"Loaded {len(logs)} evaluation logs")
    return logs

def extract_statistics(logs):
    """Extract key statistics from logs."""
    losses = []
    by_rope = defaultdict(list)
    by_goal = defaultdict(list)
    
    for log in logs:
        loss = log['loss']
        losses.append(loss)
        
        # Extract rope and goal from episode_id
        episode = log['episode_id']
        rope_idx = int(episode.split('_')[0].replace('rope', ''))
        goal_idx = int(episode.split('_')[1].replace('goal', ''))
        
        by_rope[rope_idx].append(loss)
        by_goal[goal_idx].append(loss)
    
    stats = {
        'losses': np.array(losses),
        'by_rope': {k: np.array(v) for k, v in by_rope.items()},
        'by_goal': {k: np.array(v) for k, v in by_goal.items()},
        'mean': np.mean(losses),
        'median': np.median(losses),
        'std': np.std(losses),
        'min': np.min(losses),
        'max': np.max(losses),
        'n_episodes': len(losses)
    }
    
    return stats

def plot_loss_distribution(stats, output_dir):
    """Plot loss distribution histogram."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    losses = stats['losses'] * 100  # Convert to cm
    
    ax.hist(losses, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(stats['mean']*100, color='red', linestyle='--', linewidth=2, label=f'Mean: {stats["mean"]*100:.2f} cm')
    ax.axvline(stats['median']*100, color='orange', linestyle='--', linewidth=2, label=f'Median: {stats["median"]*100:.2f} cm')
    
    ax.set_xlabel('Final Loss (cm)', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.set_title('Distribution of Final Losses across 55 Evaluation Episodes', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_distribution.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'loss_distribution.png'}")
    plt.close()

def plot_loss_by_rope(stats, output_dir):
    """Plot losses grouped by rope configuration."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rope_indices = sorted(stats['by_rope'].keys())
    rope_means = [np.mean(stats['by_rope'][i])*100 for i in rope_indices]
    rope_stds = [np.std(stats['by_rope'][i])*100 for i in rope_indices]
    
    ax.bar(rope_indices, rope_means, yerr=rope_stds, capsize=5, 
           color='steelblue', edgecolor='black', alpha=0.7)
    
    ax.set_xlabel('Rope Configuration', fontsize=14)
    ax.set_ylabel('Mean Final Loss (cm)', fontsize=14)
    ax.set_title('Performance by Rope Configuration', fontsize=16)
    ax.set_xticks(rope_indices)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_by_rope.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'loss_by_rope.png'}")
    plt.close()

def plot_loss_by_goal(stats, output_dir):
    """Plot losses grouped by goal position."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    goal_indices = sorted(stats['by_goal'].keys())
    goal_means = [np.mean(stats['by_goal'][i])*100 for i in goal_indices]
    goal_stds = [np.std(stats['by_goal'][i])*100 for i in goal_indices]
    
    ax.bar(goal_indices, goal_means, yerr=goal_stds, capsize=5,
           color='coral', edgecolor='black', alpha=0.7)
    
    ax.set_xlabel('Goal Position Index', fontsize=14)
    ax.set_ylabel('Mean Final Loss (cm)', fontsize=14)
    ax.set_title('Performance by Goal Position', fontsize=16)
    ax.set_xticks(goal_indices)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_by_goal.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'loss_by_goal.png'}")
    plt.close()

def plot_heatmap(stats, output_dir):
    """Plot heatmap of losses by rope and goal."""
    rope_indices = sorted(stats['by_rope'].keys())
    goal_indices = sorted(stats['by_goal'].keys())
    
    # Create matrix
    matrix = np.zeros((len(rope_indices), len(goal_indices)))
    
    for log_file in sorted(Path(log_dir).glob("action_log_*.json")):
        with open(log_file, 'r') as f:
            data = json.load(f)
            episode = data['episode_id']
            rope_idx = int(episode.split('_')[0].replace('rope', ''))
            goal_idx = int(episode.split('_')[1].replace('goal', ''))
            
            r = rope_indices.index(rope_idx)
            g = goal_indices.index(goal_idx)
            matrix[r, g] = data['loss'] * 100  # Convert to cm
    
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=10)
    
    ax.set_xticks(range(len(goal_indices)))
    ax.set_yticks(range(len(rope_indices)))
    ax.set_xticklabels(goal_indices)
    ax.set_yticklabels(rope_indices)
    
    ax.set_xlabel('Goal Position', fontsize=14)
    ax.set_ylabel('Rope Configuration', fontsize=14)
    ax.set_title('Heatmap: Final Loss (cm) by Rope and Goal', fontsize=16)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Loss (cm)', fontsize=12)
    
    # Add text annotations
    for i in range(len(rope_indices)):
        for j in range(len(goal_indices)):
            text = ax.text(j, i, f'{matrix[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'loss_heatmap.png'}")
    plt.close()

def create_summary_table(stats, output_dir):
    """Create summary statistics table."""
    summary = f"""
=== EVALUATION RESULTS SUMMARY ===

Total Episodes: {stats['n_episodes']}
Rope Configurations: {len(stats['by_rope'])}
Goal Positions: {len(stats['by_goal'])}

Overall Statistics:
  Mean Loss:   {stats['mean']*100:.2f} cm
  Median Loss: {stats['median']*100:.2f} cm
  Std Dev:     {stats['std']*100:.2f} cm
  Min Loss:    {stats['min']*100:.2f} cm
  Max Loss:    {stats['max']*100:.2f} cm

Performance by Rope Configuration:
"""
    for rope_idx in sorted(stats['by_rope'].keys()):
        losses = stats['by_rope'][rope_idx] * 100
        summary += f"  Rope {rope_idx}: {np.mean(losses):.2f} ± {np.std(losses):.2f} cm (n={len(losses)})\n"
    
    summary += "\nPerformance by Goal Position:\n"
    for goal_idx in sorted(stats['by_goal'].keys()):
        losses = stats['by_goal'][goal_idx] * 100
        summary += f"  Goal {goal_idx:2d}: {np.mean(losses):.2f} ± {np.std(losses):.2f} cm (n={len(losses)})\n"
    
    # Save to file
    with open(output_dir / 'summary.txt', 'w') as f:
        f.write(summary)
    
    print(summary)
    print(f"\nSaved: {output_dir / 'summary.txt'}")

if __name__ == "__main__":
    # Configuration
    log_dir = Path("outputs/2025-12-01/13-04-31/action_logs")
    output_dir = Path("eval_analysis")
    output_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("Evaluation Results Analysis for Diploma Thesis")
    print("="*60)
    
    # Load data
    logs = load_all_logs(log_dir)
    
    # Extract statistics
    stats = extract_statistics(logs)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_loss_distribution(stats, output_dir)
    plot_loss_by_rope(stats, output_dir)
    plot_loss_by_goal(stats, output_dir)
    plot_heatmap(stats, output_dir)
    
    # Create summary
    print("\n" + "="*60)
    create_summary_table(stats, output_dir)
    print("="*60)
    print(f"\nAll outputs saved to: {output_dir}/")
    print("\nFiles created:")
    for f in sorted(output_dir.glob("*")):
        print(f"  - {f.name}")
