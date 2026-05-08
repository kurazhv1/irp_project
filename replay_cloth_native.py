#!/usr/bin/env python3
"""
Replay cloth manipulation actions using native MuJoCo simulate viewer with cloth.xml.

This script launches the native MuJoCo simulate executable with the correct cloth.xml model.

Usage:
    python replay_cloth_native.py <action_log.json>

Example:
    python replay_cloth_native.py outputs/2025-10-26/16-05-10/action_logs/action_log_20251026_160517_rope0_goal0.json
"""

import os
import sys
import json
import subprocess
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: python replay_cloth_native.py <action_log.json>")
        sys.exit(1)
    
    action_log_path = sys.argv[1]
    
    if not os.path.exists(action_log_path):
        print(f"❌ Error: Action log not found: {action_log_path}")
        sys.exit(1)
    
    # Load action log to show info
    with open(action_log_path, 'r') as f:
        action_data = json.load(f)
    
    metadata = action_data['metadata']
    actions = action_data['actions']
    
    print("=" * 70)
    print("🎬 CLOTH MANIPULATION - NATIVE MUJOCO VIEWER")
    print("=" * 70)
    print(f"\n📋 Episode Info:")
    print(f"   Run ID:        {metadata['run_id']}")
    print(f"   Rope ID:       {metadata['rope_id']}")
    print(f"   Rope params:   {metadata['rope_param']}")
    print(f"   Goal alpha:    {metadata['goal_alpha']}")
    print(f"   Total steps:   {len(actions)}")
    print()
    
    # Find MuJoCo simulate executable
    mujoco_paths = [
        Path.home() / ".mujoco" / "mujoco210" / "bin" / "simulate",
        Path.home() / ".mujoco" / "mujoco200" / "bin" / "simulate",
    ]
    
    simulate_exe = None
    for path in mujoco_paths:
        if path.exists():
            simulate_exe = path
            break
    
    if not simulate_exe:
        print("❌ Error: MuJoCo simulate executable not found!")
        print("   Searched locations:")
        for path in mujoco_paths:
            print(f"   - {path}")
        sys.exit(1)
    
    # Use the original cloth.xml
    project_root = Path(__file__).parent
    cloth_xml = project_root / "assets" / "mujoco" / "cloth" / "cloth.xml"
    
    if not cloth_xml.exists():
        print(f"❌ Error: cloth.xml not found: {cloth_xml}")
        sys.exit(1)
    
    print(f"🔧 Launching MuJoCo simulate viewer...")
    print(f"   Executable: {simulate_exe}")
    print(f"   Model:      {cloth_xml}")
    print()
    print("💡 Controls in MuJoCo viewer:")
    print("   - Left mouse: Rotate view")
    print("   - Right mouse: Move view")
    print("   - Scroll: Zoom")
    print("   - Space: Pause/Resume simulation")
    print("   - Backspace: Reset simulation")
    print("   - Ctrl+C or close window to exit")
    print("=" * 70)
    print()
    
    # Launch MuJoCo simulate
    try:
        subprocess.run([str(simulate_exe), str(cloth_xml)])
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
