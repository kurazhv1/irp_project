#!/usr/bin/env python3
"""
Simple cloth visualization - just show the model without simulation.

This uses native MuJoCo simulate instead of mujoco-py to avoid segfaults.

Usage:
    python show_cloth_simple.py
"""

import sys
import subprocess
from pathlib import Path

project_root = Path(__file__).parent


def main():
    print("=" * 70)
    print("🎬 SIMPLE CLOTH VISUALIZATION")
    print("=" * 70)
    
    # Find MuJoCo simulate
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
        print("❌ Error: MuJoCo simulate not found!")
        sys.exit(1)
    
    # Use cloth.xml
    cloth_xml = project_root / "assets" / "mujoco" / "cloth" / "cloth.xml"
    
    if not cloth_xml.exists():
        print(f"❌ Error: cloth.xml not found: {cloth_xml}")
        sys.exit(1)
    
    print(f"\n📂 Model: {cloth_xml}")
    print(f"🔧 Viewer: {simulate_exe}")
    print("\n💡 Controls:")
    print("   - Left mouse: Rotate")
    print("   - Right mouse: Pan")
    print("   - Scroll: Zoom")
    print("   - Space: Play/Pause")
    print("   - Backspace: Reset")
    print("   - Double-click body: Select/follow")
    print("   - Ctrl+Right-click body: Apply force")
    print("=" * 70)
    print("\n▶️  Launching viewer...")
    
    try:
        subprocess.run([str(simulate_exe), str(cloth_xml)])
    except KeyboardInterrupt:
        print("\n👋 Closed")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
