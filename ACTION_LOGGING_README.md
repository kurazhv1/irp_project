# Action Logging and Replay System

## Overview
This system allows you to:
1. Run cloth manipulation experiments and save all actions to JSON files
2. Replay those actions on MuJoCo 3 (newer version) for comparison/visualization

## Modified Files

### `eval_irp_cloth_sim.py`
Added action logging functionality:
- Creates `action_logs/` directory in output folder
- Saves one JSON file per episode: `action_log_{run_id}_rope{rope_id}_goal{goal_id}.json`
- Each file contains:
  - Metadata (rope parameters, goal, timestamps, etc.)
  - Full sequence of actions, delta actions, losses, and control parameters

### `replay_actions_mujoco3.py` (NEW)
Script for replaying saved actions in MuJoCo 3:
- Loads action log JSON files
- Replays actions in MuJoCo 3 simulator
- Supports visualization and video recording

## Usage

### 1. Generate Action Logs (Legacy Environment)

```bash
# Activate legacy environment (mujoco-py)
conda activate irp_legacy

# Run evaluation (will create action_logs/ with JSON files)
cd /home/aalto-robotics/IRP_Project/original/irp_project
python eval_irp_cloth_sim.py

# Action logs will be saved to: outputs/{date}/{time}/action_logs/
```

### 2. Replay Actions (Modern Environment with MuJoCo 3)

```bash
# Activate modern environment (mujoco 3.x)
conda activate irp_modern

# Replay with visualization
python replay_actions_mujoco3.py outputs/2025-10-26/14-30-00/action_logs/action_log_20251026_143000_rope0_goal0.json

# Replay without visualization (headless)
python replay_actions_mujoco3.py --no-vis outputs/.../action_log_....json

# Replay and save video
python replay_actions_mujoco3.py --save-video output_replay.mp4 outputs/.../action_log_....json

# Use custom MuJoCo model
python replay_actions_mujoco3.py --model assets/mujoco/cloth/custom_cloth.xml outputs/.../action_log_....json
```

## Action Log JSON Format

```json
{
  "metadata": {
    "run_id": "20251026_143000",
    "rope_id": 0,
    "rope_param": [12.0, 1000.0],
    "goal_id": 0,
    "goal_alpha": 0.5,
    "n_steps": 50,
    "init_action": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "timestamp": "2025-10-26T14:30:00.123456"
  },
  "actions": [
    {
      "step_id": 0,
      "action": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
      "delta_action": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
      "loss": 0.5432,
      "sigma": 0.1,
      "threshold": 0.05
    },
    ...
  ]
}
```

## Data for Diploma

After running evaluations, you will have:

1. **Action Logs** (`action_logs/*.json`):
   - Complete action sequences
   - Can be analyzed, visualized, compared
   - Reproducible experiments

2. **Pickle Logs** (`log.pkl`):
   - Original format with trajectories and errors
   - Backward compatible with existing analysis code

3. **Wandb Logs**:
   - Real-time monitoring
   - Error plots and metrics

4. **Replay Videos** (optional):
   - Visual comparison between legacy and modern MuJoCo
   - Can be included in diploma presentation

## Directory Structure After Running

```
outputs/
└── 2025-10-26/
    └── 14-30-00/
        ├── action_logs/
        │   ├── action_log_20251026_143000_rope0_goal0.json
        │   ├── action_log_20251026_143000_rope0_goal1.json
        │   └── ...
        ├── config.yaml
        └── log.pkl
```

## Next Steps

1. ✅ Data unpacked
2. ✅ Action logging added
3. ✅ Replay script created
4. 🔄 Run evaluation to generate logs (current)
5. ⏳ Create architecture diagram for diploma
6. ⏳ Adapt code for MuJoCo 3
7. ⏳ Fix visualization segfault
8. ⏳ Docker containerization

## Notes

- The replay script (`replay_actions_mujoco3.py`) is a template that needs to be adapted to your specific cloth model control scheme
- Action application logic (line `data.ctrl[:] = action`) needs to match your robot/cloth control interface
- You may need to adjust rendering parameters for best video quality
