# ✅ WORKING CLOTH VISUALIZATION SOLUTION

**Date:** November 30, 2025  
**Status:** ✅ CONFIRMED WORKING

## 🎯 Problem Solved

Successfully achieved cloth.xml visualization without segfaults using modern MuJoCo Python API with GLFW.

## 🔑 Key Components

### 1. Required Software Versions
```bash
conda activate irp_legacy
pip install mujoco==2.3.7  # CRITICAL: Must use 2.3.7, NOT 3.x
pip install glfw
```

**Why MuJoCo 2.3.7?**
- MuJoCo 3.x deprecated `type="cloth"` composite (requires `type="shell"`)
- cloth.xml uses old format with `<composite type="cloth">`
- MuJoCo 2.3.7 is last version supporting legacy cloth format

### 2. Working Approach

**Modern MuJoCo Python API + GLFW rendering:**
```python
import mujoco
import glfw
import numpy as np

# Load model with MuJoCo 2.3.7 (supports cloth)
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Initialize GLFW
glfw.init()
window = glfw.create_window(1200, 900, "Cloth Viewer", None, None)
glfw.make_context_current(window)

# Create MuJoCo rendering context
cam = mujoco.MjvCamera()
opt = mujoco.MjvOption()
scene = mujoco.MjvScene(model, maxgeom=10000)
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

# Rendering loop
while not glfw.window_should_close(window):
    # Apply controls
    data.ctrl[:] = [gy, gz]
    
    # Physics step
    mujoco.mj_step(model, data)
    
    # Render
    w, h = glfw.get_framebuffer_size(window)
    viewport = mujoco.MjrRect(0, 0, w, h)
    mujoco.mjv_updateScene(model, data, opt, None, cam, 
                           mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(viewport, scene, context)
    
    glfw.swap_buffers(window)
    glfw.poll_events()
```

### 3. Why Other Approaches Failed

❌ **MuJoCo 3.x + cloth.xml:**
```
ValueError: XML Error: The "cloth" composite type is deprecated. 
Please use "shell" instead.
```

❌ **mujoco-py MjViewer.render():**
```
Segmentation fault (core dumped)
```
- Crashes when viewer.render() called during cloth simulation

❌ **mujoco-py offscreen rendering:**
```
Segmentation fault (core dumped)
```
- MjRenderContextOffscreen.render() crashes with cloth

✅ **MuJoCo 2.3.7 Python API + GLFW:**
- Direct low-level rendering
- No wrapper crashes
- Full cloth physics support
- **Tested: 1851 steps without crashes!**

## 📊 Test Results

**Test Script:** `test_cloth_glfw_viewer.py`

**Results:**
```
✅ Model loaded successfully!
   Bodies: 171
   Joints: 507
   Actuators: 2
✅ GLFW initialized
✅ GLFW window created
✅ MuJoCo rendering context created

🎬 Starting visualization...
   Controls:
   - Random gripper movements
   - Close window or press ESC to exit

[... ran for 1851 steps ...]

✅ Visualization completed! (1851 steps)
👋 Cleaning up...
```

**Performance:**
- ✅ No segfaults
- ✅ Stable simulation
- ⚠️ One warning at step 1450 (instability) but continued successfully
- ✅ Clean exit

## 📁 Working Files

**Main Script:** `test_cloth_glfw_viewer.py`
- Random gripper movements
- Full cloth physics visualization
- ESC to exit

**Model:** `assets/mujoco/cloth/cloth.xml`
- 171 bodies (13×13 cloth grid + table + gripper)
- 507 joints (cloth deformation)
- 2 actuators (gripper control)

## 🚀 Next Steps

1. **Create replay script** for action logs using this working approach
2. **Integrate with trained model** predictions
3. **Generate diploma screenshots** from visualizations
4. **Document for thesis**

## 💡 Key Insights

1. **Version matters:** MuJoCo 2.3.7 is required for cloth.xml
2. **Direct rendering works:** Bypassing high-level wrappers avoids crashes
3. **GLFW is stable:** Low-level OpenGL rendering is reliable
4. **Legacy format OK:** No need to convert cloth.xml to "shell" format

## 🔧 Installation Commands

```bash
# Activate environment
conda activate irp_legacy

# Remove newer MuJoCo if installed
pip uninstall -y mujoco

# Install correct version
pip install mujoco==2.3.7 glfw

# Verify installation
python -c "import mujoco; import glfw; print(f'MuJoCo: {mujoco.__version__}')"
```

Expected output:
```
MuJoCo: 2.3.7
```

## 📝 Usage Example

```bash
cd /home/aalto-robotics/IRP_Project/original/irp_project
conda activate irp_legacy
python test_cloth_glfw_viewer.py
```

Controls:
- **ESC** - Exit visualization
- **Close window** - Exit visualization
- Gripper moves randomly every 50-150 steps

---

**Status:** ✅ PRODUCTION READY  
**Validated:** 2025-11-30  
**Use for:** Diploma screenshots, action replay, model evaluation
