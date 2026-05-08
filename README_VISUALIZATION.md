# 🎨 Cloth Visualization Guide

## ✅ Working Solution

Successfully achieved **stable cloth.xml visualization** using MuJoCo 2.3.7 + GLFW.

**Status:** Production ready for diploma screenshots and model evaluation.

---

## 🚀 Quick Start

### 1. Installation

```bash
# Activate environment
conda activate irp_legacy

# Install correct MuJoCo version (CRITICAL!)
pip uninstall -y mujoco  # Remove newer versions
pip install mujoco==2.3.7 glfw

# Verify
python -c "import mujoco; print(f'MuJoCo: {mujoco.__version__}')"
# Expected: MuJoCo: 2.3.7
```

### 2. Test Visualization

```bash
# Random gripper movements test
python test_cloth_glfw_viewer.py
```

**Controls:**
- **ESC** - Exit
- **Close window** - Exit

### 3. Replay Trained Model

```bash
# Find available action logs
ls output/*/action_log_*.json

# Replay specific episode
python replay_cloth_trained_model.py output/20241130_123456/action_log_rope0_goal5.json
```

---

## 📁 Files

| File | Purpose | Status |
|------|---------|--------|
| `test_cloth_glfw_viewer.py` | Random movement test | ✅ Working |
| `replay_cloth_trained_model.py` | Replay action logs | ✅ Ready |
| `WORKING_VISUALIZATION_SOLUTION.md` | Technical documentation | ✅ Complete |
| `assets/mujoco/cloth/cloth.xml` | Static cloth model | ✅ Production |
| `assets/mujoco/cloth/table_cloth_template.xml.jinja2` | Dynamic template | ✅ Production |

---

## 🎯 Use Cases

### For Diploma Screenshots

1. **Static visualization:**
   ```bash
   python test_cloth_glfw_viewer.py
   # Take screenshots of cloth physics
   ```

2. **Trained model results:**
   ```bash
   python replay_cloth_trained_model.py output/.../action_log_rope0_goal5.json
   # Screenshot of actual predictions
   ```

### For Model Evaluation

```bash
# Run evaluation with action logging
python eval_irp_cloth_sim.py

# Replay specific episodes
python replay_cloth_trained_model.py output/.../action_log_*.json
```

---

## 🔧 Technical Details

### Why MuJoCo 2.3.7?

- **MuJoCo 3.x:** Deprecated `type="cloth"` → requires `type="shell"`
- **cloth.xml:** Uses legacy `<composite type="cloth">` format
- **MuJoCo 2.3.7:** Last version supporting legacy cloth

### Architecture

```python
import mujoco  # 2.3.7 - supports cloth
import glfw     # OpenGL window management

# Load model
model = mujoco.MjModel.from_xml_path("cloth.xml")
data = mujoco.MjData(model)

# Create rendering context
window = glfw.create_window(...)
scene = mujoco.MjvScene(model, maxgeom=10000)
context = mujoco.MjrContext(model, ...)

# Render loop
while not glfw.window_should_close(window):
    mujoco.mj_step(model, data)  # Physics
    mujoco.mjv_updateScene(...)   # Update scene
    mujoco.mjr_render(...)        # Render
    glfw.swap_buffers(window)
```

### Why Not mujoco-py?

❌ **mujoco-py MjViewer:** Segfaults during cloth simulation  
❌ **Offscreen rendering:** Crashes with cloth physics  
✅ **Direct GLFW rendering:** Stable, no crashes

---

## 📊 Test Results

**Test:** `test_cloth_glfw_viewer.py`

```
✅ Model loaded: 171 bodies, 507 joints, 2 actuators
✅ GLFW window created
✅ Ran 1851 steps without crashes
⚠️ One instability warning (step 1450) but continued
✅ Clean exit
```

**Performance:** ~60 FPS, smooth cloth deformation

---

## 🐛 Troubleshooting

### Error: "cloth composite type is deprecated"

**Problem:** Using MuJoCo 3.x  
**Solution:** 
```bash
pip uninstall -y mujoco
pip install mujoco==2.3.7
```

### Error: "No module named 'glfw'"

**Solution:**
```bash
pip install glfw
```

### Segmentation fault

**Problem:** Using mujoco-py viewer or offscreen rendering  
**Solution:** Use `test_cloth_glfw_viewer.py` or `replay_cloth_trained_model.py`

### Window doesn't open

**Problem:** GLFW initialization failed  
**Solution:** Check X11/Wayland display:
```bash
echo $DISPLAY
# If empty: export DISPLAY=:0
```

---

## 📝 Next Steps

- [x] Working visualization with random movements
- [x] Replay script for action logs
- [ ] Integration with live model predictions
- [ ] Video recording for diploma
- [ ] Comparison visualizations (predicted vs actual)

---

## 📚 References

- **Technical Details:** `WORKING_VISUALIZATION_SOLUTION.md`
- **Original Issue:** mujoco-py viewer segfaults with cloth
- **Solution:** MuJoCo 2.3.7 Python API + GLFW direct rendering
- **Tested:** 2025-11-30, confirmed stable

---

**Status:** ✅ PRODUCTION READY  
**For:** Diploma screenshots, action replay, model evaluation  
**Maintainer:** IRP Project Team
