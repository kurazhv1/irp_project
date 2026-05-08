#!/usr/bin/env python3
"""
Simplified visualization for best episode using logged data directly.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from matplotlib.patches import Patch

# Configuration
LOG_FILE = "outputs/2025-12-01/13-04-31/action_logs/action_log_20251201_130443_rope4_goal5.json"
OUTPUT_DIR = Path("best_episode_visualization")
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*70)
print("BEST EPISODE VISUALIZATION (1.57 cm loss)")
print("="*70)

# Load data
print(f"\n📂 Loading: {LOG_FILE}")
with open(LOG_FILE, 'r') as f:
    log_data = json.load(f)

episode_id = log_data['episode_id']
loss = log_data['loss']
# raw_actions = [duration, gy1, gz1, gy2]
# Actions are trajectory parameters in YZ plane, not 3D positions
# For visualization, we'll show the trajectory control points
raw_actions = np.array(log_data['raw_actions'])
goal_coords = np.array(log_data['goal_coords'])

# Create trajectory waypoints from actions
# Each action defines: start (0,0) -> waypoint1 (gy1, gz1) -> waypoint2 (gy2, gz2=0.05)
# NOTE: Actions are in YZ plane, we swap to match goal_coords (X,Y,Z) format
trajectory_points = []
for raw_act in raw_actions:
    duration, gy1, gz1, gy2 = raw_act
    gz2 = 0.05
    # Swap: gy1->Y, gz1->Z becomes X,Y,Z with gz1->Y, gy1->Z for consistency with goal
    trajectory_points.append([0, gz1, gy1])  # [X=0, Y=gz1, Z=gy1]
    trajectory_points.append([0, gz2, gy2])  # [X=0, Y=gz2, Z=gy2]

actions = np.array(trajectory_points)

print(f"Episode: {episode_id}")
print(f"Final Loss: {loss*100:.2f} cm")
print(f"Raw actions: {len(raw_actions)} (duration, gy1, gz1, gy2)")
print(f"Trajectory waypoints: {len(actions)} points in YZ plane")
print(f"Goal coords: {goal_coords.shape} (3D cloth positions)")

# ============================================================================
# 1. 3D TRAJECTORY + GOAL POINTS
# ============================================================================
print("\n" + "="*70)
print("1. Creating 3D Trajectory with Goal Points...")

fig = plt.figure(figsize=(16, 12))

# Main 3D plot with trajectory and goal
ax1 = fig.add_subplot(221, projection='3d')
ax1.plot(actions[:, 0], actions[:, 1], actions[:, 2], 
         'b-', linewidth=2.5, alpha=0.7, label='Robot trajectory', zorder=1)
ax1.scatter(actions[0, 0], actions[0, 1], actions[0, 2], 
           c='green', s=300, marker='o', label='Start', edgecolors='black', linewidths=2, zorder=5)
ax1.scatter(actions[-1, 0], actions[-1, 1], actions[-1, 2], 
           c='red', s=300, marker='X', label='End', edgecolors='black', linewidths=2, zorder=5)

# Plot goal points
ax1.scatter(goal_coords[:, 0], goal_coords[:, 1], goal_coords[:, 2],
           c='orange', s=150, marker='s', alpha=0.8, label='Goal points (9)', 
           edgecolors='black', linewidths=1, zorder=3)

ax1.set_xlabel('X (m)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
ax1.set_zlabel('Z (m)', fontsize=12, fontweight='bold')
ax1.set_title(f'3D View: Trajectory + Goal\nLoss: {loss*100:.2f} cm', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(True, alpha=0.3)

# Top view (X-Y)
ax2 = fig.add_subplot(222)
ax2.plot(actions[:, 0], actions[:, 1], 'b-', linewidth=2.5, alpha=0.7, label='Trajectory', zorder=1)
ax2.scatter(actions[0, 0], actions[0, 1], c='green', s=300, marker='o', 
           label='Start', edgecolors='black', linewidths=2, zorder=5)
ax2.scatter(actions[-1, 0], actions[-1, 1], c='red', s=300, marker='X', 
           label='End', edgecolors='black', linewidths=2, zorder=5)
ax2.scatter(goal_coords[:, 0], goal_coords[:, 1], c='orange', s=150, 
           marker='s', alpha=0.8, label='Goal points', edgecolors='black', linewidths=1, zorder=3)

# Number waypoints
for i in range(0, len(actions), max(1, len(actions)//8)):
    ax2.annotate(f'{i}', (actions[i, 0], actions[i, 1]), 
                fontsize=9, ha='center', va='center', fontweight='bold',
                bbox=dict(boxstyle='circle', facecolor='white', alpha=0.7, edgecolor='blue'))

ax2.set_xlabel('X (m)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
ax2.set_title('Top View (X-Y Plane)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal', adjustable='box')

# Side view (X-Z)
ax3 = fig.add_subplot(223)
ax3.plot(actions[:, 0], actions[:, 2], 'b-', linewidth=2.5, alpha=0.7, label='Trajectory', zorder=1)
ax3.scatter(actions[0, 0], actions[0, 2], c='green', s=300, marker='o', 
           label='Start', edgecolors='black', linewidths=2, zorder=5)
ax3.scatter(actions[-1, 0], actions[-1, 2], c='red', s=300, marker='X', 
           label='End', edgecolors='black', linewidths=2, zorder=5)
ax3.scatter(goal_coords[:, 0], goal_coords[:, 2], c='orange', s=150, 
           marker='s', alpha=0.8, label='Goal points', edgecolors='black', linewidths=1, zorder=3)

ax3.set_xlabel('X (m)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Z (m)', fontsize=12, fontweight='bold')
ax3.set_title('Side View (X-Z Plane)', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# Front view (Y-Z)
ax4 = fig.add_subplot(224)
ax4.plot(actions[:, 1], actions[:, 2], 'b-', linewidth=2.5, alpha=0.7, label='Trajectory', zorder=1)
ax4.scatter(actions[0, 1], actions[0, 2], c='green', s=300, marker='o', 
           label='Start', edgecolors='black', linewidths=2, zorder=5)
ax4.scatter(actions[-1, 1], actions[-1, 2], c='red', s=300, marker='X', 
           label='End', edgecolors='black', linewidths=2, zorder=5)
ax4.scatter(goal_coords[:, 1], goal_coords[:, 2], c='orange', s=150, 
           marker='s', alpha=0.8, label='Goal points', edgecolors='black', linewidths=1, zorder=3)

ax4.set_xlabel('Y (m)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Z (m)', fontsize=12, fontweight='bold')
ax4.set_title('Front View (Y-Z Plane)', fontsize=14, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
trajectory_file = OUTPUT_DIR / "01_trajectory_with_goals.png"
plt.savefig(trajectory_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {trajectory_file}")
plt.close()

# ============================================================================
# 2. GOAL POINTS GRID VISUALIZATION
# ============================================================================
print("\n" + "="*70)
print("2. Creating Goal Points Grid Visualization...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Goal points in 2D grid (top view)
ax = axes[0]
# Reshape goal_coords to 3x3 grid
goal_grid = goal_coords.reshape(3, 3, 3)  # 3x3 grid, 3 coords (x,y,z)

# Plot grid structure
for i in range(3):
    for j in range(3):
        x, y = goal_grid[i, j, 0], goal_grid[i, j, 1]
        ax.scatter(x, y, c='orange', s=400, marker='s', 
                  edgecolors='black', linewidths=2, zorder=5)
        ax.text(x, y, f'{i},{j}', ha='center', va='center', 
               fontsize=10, fontweight='bold')

# Connect grid points
for i in range(3):
    for j in range(3):
        if j < 2:  # Connect horizontally
            x1, y1 = goal_grid[i, j, 0], goal_grid[i, j, 1]
            x2, y2 = goal_grid[i, j+1, 0], goal_grid[i, j+1, 1]
            ax.plot([x1, x2], [y1, y2], 'gray', linestyle='--', linewidth=1.5, alpha=0.5, zorder=1)
        if i < 2:  # Connect vertically
            x1, y1 = goal_grid[i, j, 0], goal_grid[i, j, 1]
            x2, y2 = goal_grid[i+1, j, 0], goal_grid[i+1, j, 1]
            ax.plot([x1, x2], [y1, y2], 'gray', linestyle='--', linewidth=1.5, alpha=0.5, zorder=1)

ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
ax.set_title('Goal Grid (Top View)\n3×3 cloth points', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal', adjustable='box')

# Goal points in 3D
ax2 = fig.add_subplot(122, projection='3d')
for i in range(3):
    for j in range(3):
        x, y, z = goal_grid[i, j, 0], goal_grid[i, j, 1], goal_grid[i, j, 2]
        ax2.scatter(x, y, z, c='orange', s=400, marker='s', 
                   edgecolors='black', linewidths=2, zorder=5)
        ax2.text(x, y, z, f'  {i},{j}', fontsize=8)

# Connect grid points in 3D
for i in range(3):
    for j in range(3):
        if j < 2:
            p1 = goal_grid[i, j]
            p2 = goal_grid[i, j+1]
            ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                    'gray', linestyle='--', linewidth=1.5, alpha=0.5, zorder=1)
        if i < 2:
            p1 = goal_grid[i, j]
            p2 = goal_grid[i+1, j]
            ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                    'gray', linestyle='--', linewidth=1.5, alpha=0.5, zorder=1)

ax2.set_xlabel('X (m)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
ax2.set_zlabel('Z (m)', fontsize=12, fontweight='bold')
ax2.set_title('Goal Grid (3D View)', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
grid_file = OUTPUT_DIR / "02_goal_grid.png"
plt.savefig(grid_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {grid_file}")
plt.close()

# ============================================================================
# 3. ACTION ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("3. Creating Action Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Action components over time
ax = axes[0, 0]
time_steps = np.arange(len(actions))
ax.plot(time_steps, actions[:, 0], 'r-', linewidth=2.5, label='X', marker='o')
ax.plot(time_steps, actions[:, 1], 'g-', linewidth=2.5, label='Y', marker='s')
ax.plot(time_steps, actions[:, 2], 'b-', linewidth=2.5, label='Z', marker='^')
ax.set_xlabel('Action Step', fontsize=12, fontweight='bold')
ax.set_ylabel('Position (m)', fontsize=12, fontweight='bold')
ax.set_title('End-Effector Position Components', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Action deltas (movement magnitude)
ax = axes[0, 1]
deltas = np.diff(actions, axis=0)
magnitudes = np.linalg.norm(deltas, axis=1)
ax.bar(time_steps[:-1], magnitudes, color='purple', alpha=0.7, edgecolor='black')
ax.set_xlabel('Action Step', fontsize=12, fontweight='bold')
ax.set_ylabel('Movement Magnitude (m)', fontsize=12, fontweight='bold')
ax.set_title('Movement per Step', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.text(0.5, 0.95, f'Mean: {np.mean(magnitudes):.4f} m\nMax: {np.max(magnitudes):.4f} m', 
        transform=ax.transAxes, ha='center', va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# Cumulative distance
ax = axes[1, 0]
cumulative = np.concatenate([[0], np.cumsum(magnitudes)])
ax.plot(time_steps, cumulative, 'darkblue', linewidth=3, marker='o', markersize=6)
ax.fill_between(time_steps, cumulative, alpha=0.3, color='lightblue')
ax.set_xlabel('Action Step', fontsize=12, fontweight='bold')
ax.set_ylabel('Cumulative Distance (m)', fontsize=12, fontweight='bold')
ax.set_title('Total Path Length', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.text(0.5, 0.95, f'Total: {cumulative[-1]:.3f} m', 
        transform=ax.transAxes, ha='center', va='top', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# Distance to goal center
ax = axes[1, 1]
goal_center = np.mean(goal_coords, axis=0)
distances = np.linalg.norm(actions - goal_center, axis=1)
ax.plot(time_steps, distances, 'darkgreen', linewidth=3, marker='s', markersize=6)
ax.axhline(y=loss*100/100, color='red', linestyle='--', linewidth=2, label=f'Final loss: {loss*100:.2f} cm')
ax.set_xlabel('Action Step', fontsize=12, fontweight='bold')
ax.set_ylabel('Distance to Goal Center (m)', fontsize=12, fontweight='bold')
ax.set_title('Distance to Goal During Execution', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
analysis_file = OUTPUT_DIR / "03_action_analysis.png"
plt.savefig(analysis_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {analysis_file}")
plt.close()

# ============================================================================
# 4. SUMMARY TABLE
# ============================================================================
print("\n" + "="*70)
print("4. Creating Summary...")

summary = f"""
{'='*70}
BEST EPISODE SUMMARY
{'='*70}

Episode ID: {episode_id}
Final Loss: {loss*100:.2f} cm  (BEST out of 55 episodes)

Trajectory Statistics:
  Total Actions: {len(actions)}
  Total Path Length: {cumulative[-1]:.3f} m
  Mean Step Size: {np.mean(magnitudes):.4f} m
  Max Step Size: {np.max(magnitudes):.4f} m
  Min Step Size: {np.min(magnitudes):.4f} m

Goal Configuration:
  Number of Points: {len(goal_coords)} (3×3 grid)
  Goal Center: ({goal_center[0]:.3f}, {goal_center[1]:.3f}, {goal_center[2]:.3f})
  Goal X range: [{goal_coords[:, 0].min():.3f}, {goal_coords[:, 0].max():.3f}]
  Goal Y range: [{goal_coords[:, 1].min():.3f}, {goal_coords[:, 1].max():.3f}]
  Goal Z range: [{goal_coords[:, 2].min():.3f}, {goal_coords[:, 2].max():.3f}]

Start Position: ({actions[0, 0]:.3f}, {actions[0, 1]:.3f}, {actions[0, 2]:.3f})
End Position: ({actions[-1, 0]:.3f}, {actions[-1, 1]:.3f}, {actions[-1, 2]:.3f})

Final Distance to Goal Center: {distances[-1]:.4f} m

{'='*70}
"""

with open(OUTPUT_DIR / "summary.txt", 'w') as f:
    f.write(summary)

print(summary)
print(f"✓ Saved: {OUTPUT_DIR / 'summary.txt'}")

# ============================================================================
# FINAL
# ============================================================================
print("\n" + "="*70)
print("VISUALIZATION COMPLETE!")
print("="*70)
print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
print("\nGenerated files:")
for f in sorted(OUTPUT_DIR.glob("*")):
    size = f.stat().st_size / 1024
    print(f"  ✓ {f.name} ({size:.1f} KB)")
print("="*70)
