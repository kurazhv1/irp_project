# %%
# import
import os
import pathlib
import click
import yaml
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import zarr
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import wandb
from scipy.interpolate import interp1d
import mujoco_py as mj
from hydra.utils import to_absolute_path
import json
from datetime import datetime

from common.sample_util import transpose_data_dict
from environments.table_cloth_sim_environment import TableClothSimEnvironment
from common.wandb_util import get_error_plots_log

from networks.cloth_delta_deeplab import ClothDeltaDeeplab
from real_ur5.delta_action_sampler import DeltaActionGaussianSampler
from real_ur5.delta_action_selector import DeltaActionLossSelector


# %%
@hydra.main(config_path="config", config_name=pathlib.Path(__file__).stem)
def main(cfg: DictConfig) -> None:
    wandb.init(**cfg.wandb)
    config = OmegaConf.to_container(cfg, resolve=True)
    output_dir = os.getcwd()
    config['output_dir'] = output_dir
    yaml.dump(config, open('config.yaml', 'w'), default_flow_style=False)
    wandb.config.update(config)

    # Create directory for action logs
    action_log_dir = pathlib.Path(output_dir) / "action_logs"
    action_log_dir.mkdir(exist_ok=True)
    
    # Generate run ID for this evaluation
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # load action model
    device = torch.device('cuda', cfg.action.gpu_id)
    dtype = torch.float16 if cfg.action.use_fp16 else torch.float32
    sampler = DeltaActionGaussianSampler(**cfg.action.sampler)
    action_model = ClothDeltaDeeplab.load_from_checkpoint(
        to_absolute_path(cfg.action.ckpt_path))
    action_model_gpu = action_model.to(
        device, dtype=dtype).eval()
    selector = DeltaActionLossSelector(
        model=action_model_gpu, **cfg.action.selector)

    # action
    ropes_log = list()
    for rope_id, rope_param in enumerate(cfg.setup.selection.cloth_size_density):
        this_rope_cfg = OmegaConf.to_container(cfg.setup.rope_config, resolve=True)
        this_rope_cfg['cloth_spacing'] = rope_param[0] / 12
        this_rope_cfg['cloth_density'] = rope_param[1]
        
        # Enable visualization if specified in config
        show_vis = cfg.setup.get('show_vis', False)
        env = TableClothSimEnvironment(this_rope_cfg, cfg.setup.controller_config,
            obs_topdown=cfg.setup.obs_topdown, show_vis=show_vis)

        goals_log = list()
        for goal_id, goal_alpha in enumerate(cfg.setup.selection.goal_alpha):
            goal = env.get_cloth_goal(goal_alpha)
            loss_func = env.get_traj_loss_func(goal, **cfg.setup.traj_loss)
            img_loss_func = env.get_img_loss_func(goal, **cfg.setup.img_loss)
            env.set_loss_func(loss_func)

            # init action
            init_action = np.array(cfg.action.init_action)
            action = init_action

            # Action log for this episode
            episode_actions = []
            episode_raw_actions = []  # For replay visualization
            
            # Get goal coords for this episode
            goal_coords = env.get_cloth_goal(goal_alpha)
            
            episode_metadata = {
                'run_id': run_id,
                'rope_id': int(rope_id),
                'rope_param': [float(x) for x in rope_param],
                'goal_id': int(goal_id),
                'goal_alpha': float(goal_alpha),
                'n_steps': int(cfg.setup.n_steps),
                'init_action': [float(x) for x in init_action],
                'timestamp': datetime.now().isoformat()
            }

            steps_log = list()
            for step_id in tqdm(range(cfg.setup.n_steps)):
                try:
                    observation, loss, _, info = env.step(action)
                except mj.MujocoException:
                    print(goal_id, step_id, action)
                    pass

                sigma = cfg.action.constant_sigma
                if sigma is None:
                    sigma = min(loss * cfg.action.gain, cfg.action.sigma_max)
                
                ts = cfg.action.threshold
                threshold_interp = interp1d(
                    x=[0, ts.dist_max], y=[ts.max, ts.min], 
                    kind='linear',
                    bounds_error=False,
                    fill_value=(ts.max, ts.min))
                threshold = threshold_interp(min(loss, ts.dist_max))

                delta_action_samples = sampler.get_delta_action_samples(
                    action, sigma=sigma)
                selection_result = selector.get_delta_action(
                    traj_img=observation, 
                    delta_action_samples=delta_action_samples,
                    loss_func=img_loss_func,
                    threshold=threshold)
                best_delta_action = selection_result['best_delta_action']

                # Save action for this step
                episode_actions.append({
                    'step_id': int(step_id),
                    'action': [float(x) for x in action],
                    'delta_action': [float(x) for x in best_delta_action],
                    'loss': float(loss),
                    'sigma': float(sigma),
                    'threshold': float(threshold)
                })
                
                # Save raw action (mapped from normalized action) for visualization
                raw_action = env.action_mapper(action)  # Returns [duration, gy1, gz1, gy2]
                episode_raw_actions.append([float(x) for x in raw_action])

                # logging
                steps_log.append({
                    'error': loss,
                    'action': action,
                    'trajectory': info['trajectory']
                })
                # next
                action = best_delta_action + action

            # Save action log for this episode (format compatible with replay script)
            action_log = {
                'episode_id': f"{run_id}_rope{rope_id}_goal{goal_id}",
                'timestamp': run_id,
                'rope_config': this_rope_cfg,
                'controller_config': OmegaConf.to_container(cfg.setup.controller_config, resolve=True),
                'goal_alpha': float(goal_alpha),
                'goal_coords': goal_coords.tolist(),
                'actions': [[float(x) for x in ep['action']] for ep in episode_actions],
                'raw_actions': episode_raw_actions,
                'loss': float(loss),  # Final loss
                'metadata': episode_metadata,
                'detailed_actions': episode_actions  # Full step-by-step data
            }
            log_filename = action_log_dir / f"action_log_{run_id}_rope{rope_id}_goal{goal_id}.json"
            with open(log_filename, 'w') as f:
                json.dump(action_log, f, indent=2)
            print(f"✅ Saved action log: {log_filename}")

            # aggregate
            steps_log = transpose_data_dict(steps_log)
            print('Min error:', steps_log['error'].min())
            goals_log.append(steps_log)
        
        # aggregate data
        rope_log = transpose_data_dict(goals_log)
        rope_key = 'cloth_' + '_'.join('{:.02f}'.format(x) for x in rope_param)
        log = get_error_plots_log(rope_key, rope_log['error'])
        wandb.log(log)
        ropes_log.append(rope_log)
    all_logs = transpose_data_dict(ropes_log)
    errors = all_logs['error'].reshape(-1, all_logs['error'].shape[-1])
    log = get_error_plots_log('all', errors)
    wandb.log(log)
    import pickle
    pickle.dump(all_logs, open('log.pkl', 'wb'))

    print(f"\nAll action logs saved to: {action_log_dir}")
    print(f"Total episodes: {len(list(action_log_dir.glob('*.json')))}")


# %%
if __name__ == '__main__':
    main()

# %%
