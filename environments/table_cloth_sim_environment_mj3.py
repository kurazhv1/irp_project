"""
TableClothSimEnvironment adapted for MuJoCo 3
Provides modern API interface for cloth manipulation simulation
"""

import pathlib
import numpy as np
from typing import Tuple, Optional, Dict, Any
from jinja2 import Template

try:
    import mujoco
    MUJOCO3_AVAILABLE = True
except ImportError:
    MUJOCO3_AVAILABLE = False
    print("Warning: MuJoCo 3 not available")

from common.cv_util import get_traj_occupancy
from common.sample_util import GridCoordTransformer, get_nd_index_volume


class TableClothSimEnvironmentMJ3:
    """
    Cloth manipulation environment using MuJoCo 3
    
    This is a modern adaptation of the legacy mujoco-py based environment.
    Key differences:
    - Uses mujoco (MuJoCo 3+) instead of mujoco_py
    - Cleaner API with mujoco.MjModel and mujoco.MjData
    - Better rendering support
    - More stable cloth simulation
    """
    
    def __init__(
        self,
        rope_config: Dict[str, float],
        controller_config: Dict[str, Any],
        obs_topdown: bool = False,
        show_vis: bool = False,
        xml_path: Optional[str] = None
    ):
        """
        Initialize cloth simulation environment
        
        Args:
            rope_config: Configuration for cloth/rope properties
                - table_height: Height of table surface
                - table_y: Y position of table
                - table_size: Size of table
                - cloth_spacing: Spacing between cloth nodes
                - cloth_density: Density of cloth material
            controller_config: PD controller configuration
                - joint_names: List of joint names to control
                - kp: Proportional gain
                - kv: Derivative gain
            obs_topdown: Whether to use top-down camera view
            show_vis: Whether to enable live visualization
            xml_path: Optional path to custom XML model
        """
        if not MUJOCO3_AVAILABLE:
            raise ImportError("MuJoCo 3 is required. Install with: pip install mujoco")
        
        self.rope_config = rope_config
        self.controller_config = controller_config
        self.obs_topdown = obs_topdown
        self.show_vis = show_vis
        
        # Generate or load XML model
        if xml_path is None:
            xml_path = self._generate_xml_from_config(rope_config)
        
        self.xml_path = pathlib.Path(xml_path)
        if not self.xml_path.exists():
            raise FileNotFoundError(f"XML model not found: {xml_path}")
        
        # Load MuJoCo model
        print(f"Loading MuJoCo model: {self.xml_path}")
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        
        # Renderer for observations
        self.renderer = mujoco.Renderer(self.model, height=256, width=256)
        
        # Viewer for live visualization
        self.viewer = None
        if show_vis:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        # Camera configuration
        self.camera_name = "topdown" if obs_topdown else "fixed"
        self.camera_id = mujoco.mj_name2id(
            self.model, 
            mujoco.mjtObj.mjOBJ_CAMERA, 
            self.camera_name
        )
        
        # Get cloth body IDs
        self.cloth_body_ids = self._get_cloth_body_ids()
        
        # Loss function
        self.loss_func = None
        
        # State
        self.current_step = 0
        self.max_steps = 16
        
        print("✓ Environment initialized")
        print(f"  - Bodies: {self.model.nbody}")
        print(f"  - Joints: {self.model.njnt}")
        print(f"  - Actuators: {self.model.nu}")
        print(f"  - Cloth bodies: {len(self.cloth_body_ids)}")
    
    def _generate_xml_from_config(self, config: Dict) -> str:
        """Generate XML file from configuration"""
        # Use MuJoCo 3 compatible XML
        default_xml_mj3 = pathlib.Path("assets/mujoco/cloth/cloth.xml")
        if default_xml_mj3.exists():
            print(f"Using MuJoCo 3 compatible XML: {default_xml_mj3}")
            return str(default_xml_mj3)
        
        # Fallback to template (may need adaptation for MJ3)
        template_path = pathlib.Path("assets/mujoco/cloth/table_cloth_template.xml.jinja2")
        
        if not template_path.exists():
            # Last resort - original cloth.xml (may not work with MJ3)
            default_xml = pathlib.Path("assets/mujoco/cloth/cloth.xml")
            if default_xml.exists():
                print(f"Warning: Using legacy XML, may not be compatible with MuJoCo 3")
                return str(default_xml)
            else:
                raise FileNotFoundError("No cloth XML template found")
        
        with open(template_path, 'r') as f:
            template = Template(f.read())
        
        xml_content = template.render(
            table_height=config.get('table_height', 0.8),
            table_y=config.get('table_y', 1.0),
            table_size=config.get('table_size', 1.2),
            cloth_spacing=config.get('cloth_spacing', 0.0383),
            cloth_density=config.get('cloth_density', 0.98)
        )
        
        # Save to temporary file
        output_path = pathlib.Path("generated_cloth_temp.xml")
        with open(output_path, 'w') as f:
            f.write(xml_content)
        
        return str(output_path)
    
    def _get_cloth_body_ids(self) -> list:
        """Get body IDs for all cloth elements"""
        cloth_ids = []
        for i in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name and 'cloth' in body_name.lower():
                cloth_ids.append(i)
        return cloth_ids
    
    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state
        
        Returns:
            Initial observation (RGB image)
        """
        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0
        
        # Forward simulation to settle
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)
        
        return self.get_obs()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one environment step
        
        Args:
            action: Action vector [x, y, theta, pull_length]
                - x, y: Normalized grasp position (0-1)
                - theta: Pull direction in radians
                - pull_length: Distance to pull
        
        Returns:
            observation: RGB image (256, 256, 3)
            loss: Distance to goal configuration
            done: Whether episode is finished
            info: Additional information
        """
        # Apply action to simulation
        self._apply_action(action)
        
        # Step physics simulation
        for _ in range(10):  # Multiple substeps per action
            mujoco.mj_step(self.model, self.data)
            
            # Update viewer if enabled
            if self.viewer is not None:
                self.viewer.sync()
        
        # Get observation and compute loss
        obs = self.get_obs()
        loss = self.compute_loss() if self.loss_func else 0.0
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        info = {
            'step': self.current_step,
            'cloth_positions': self.get_cloth_positions()
        }
        
        return obs, loss, done, info
    
    def _apply_action(self, action: np.ndarray):
        """
        Apply action to robot/cloth
        
        This is a simplified version - needs to be adapted based on
        your specific robot control scheme.
        """
        # TODO: Implement proper action application
        # For now, just apply to control vector
        if len(action) <= self.model.nu:
            self.data.ctrl[:len(action)] = action
    
    def get_obs(self) -> np.ndarray:
        """
        Get current observation (RGB image)
        
        Returns:
            RGB image array (256, 256, 3) in range [0, 255]
        """
        # Update renderer
        self.renderer.update_scene(self.data, camera=self.camera_id)
        
        # Render frame
        rgb = self.renderer.render()
        
        # Ensure correct format
        if rgb.shape != (256, 256, 3):
            import cv2
            rgb = cv2.resize(rgb, (256, 256))
        
        return rgb
    
    def get_cloth_positions(self) -> np.ndarray:
        """
        Get current positions of all cloth bodies
        
        Returns:
            Array of shape (N, 3) with XYZ positions
        """
        positions = []
        for body_id in self.cloth_body_ids:
            pos = self.data.xpos[body_id]
            positions.append(pos)
        return np.array(positions)
    
    def get_cloth_goal(self, alpha: float) -> np.ndarray:
        """
        Get target cloth configuration for given alpha
        
        Args:
            alpha: Goal parameter in [0, 1]
        
        Returns:
            Target cloth positions
        """
        # TODO: Implement goal generation based on alpha
        # This depends on your specific task setup
        # For now, return current positions
        return self.get_cloth_positions()
    
    def set_loss_func(self, loss_func):
        """Set loss/reward function"""
        self.loss_func = loss_func
    
    def get_traj_loss_func(self, goal_positions: np.ndarray, measure_dims: list = [0, 1, 2]):
        """
        Create trajectory loss function
        
        Args:
            goal_positions: Target cloth positions
            measure_dims: Dimensions to measure (0=x, 1=y, 2=z)
        
        Returns:
            Loss function
        """
        def loss_func():
            current_pos = self.get_cloth_positions()
            
            # Compute L2 distance in specified dimensions
            diff = current_pos[:, measure_dims] - goal_positions[:, measure_dims]
            loss = np.sqrt(np.sum(diff ** 2))
            
            return loss
        
        return loss_func
    
    def compute_loss(self) -> float:
        """
        Compute current loss/reward
        
        Returns:
            Loss value (lower is better)
        """
        if self.loss_func is None:
            return 0.0
        
        try:
            return self.loss_func()
        except Exception as e:
            print(f"Warning: Loss computation failed: {e}")
            return 0.0
    
    def render(self, width: int = 640, height: int = 480) -> np.ndarray:
        """
        Render high-resolution frame
        
        Args:
            width: Frame width
            height: Frame height
        
        Returns:
            RGB image array
        """
        # Create temporary renderer with custom resolution
        renderer = mujoco.Renderer(self.model, height=height, width=width)
        renderer.update_scene(self.data, camera=self.camera_id)
        frame = renderer.render()
        return frame
    
    def close(self):
        """Clean up resources"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        
        # Clean up temporary XML if generated
        temp_xml = pathlib.Path("generated_cloth_temp.xml")
        if temp_xml.exists():
            temp_xml.unlink()
    
    def __del__(self):
        """Destructor"""
        self.close()


def create_environment_from_metadata(metadata: Dict) -> TableClothSimEnvironmentMJ3:
    """
    Create environment from action log metadata
    
    Args:
        metadata: Action log metadata dictionary
    
    Returns:
        Configured environment
    """
    rope_param = metadata['rope_param']
    
    rope_config = {
        'table_height': 0.8,
        'table_y': 1.0,
        'table_size': 1.2,
        'cloth_spacing': rope_param[0] / 12,
        'cloth_density': rope_param[1],
    }
    
    controller_config = {
        'joint_names': ['gy', 'gz'],
        'kp': 100000,
        'kv': 100000
    }
    
    env = TableClothSimEnvironmentMJ3(
        rope_config=rope_config,
        controller_config=controller_config,
        obs_topdown=False,
        show_vis=False
    )
    
    # Set up goal
    goal_alpha = metadata.get('goal_alpha', 0.0)
    goal = env.get_cloth_goal(goal_alpha)
    loss_func = env.get_traj_loss_func(goal, measure_dims=[0, 1, 2])
    env.set_loss_func(loss_func)
    
    return env


# Example usage
if __name__ == "__main__":
    print("Testing TableClothSimEnvironmentMJ3")
    
    # Create simple test environment
    rope_config = {
        'table_height': 0.8,
        'table_y': 1.0,
        'table_size': 1.2,
        'cloth_spacing': 0.0383,
        'cloth_density': 0.98,
    }
    
    controller_config = {
        'joint_names': ['gy', 'gz'],
        'kp': 100000,
        'kv': 100000
    }
    
    try:
        env = TableClothSimEnvironmentMJ3(
            rope_config=rope_config,
            controller_config=controller_config,
            obs_topdown=False,
            show_vis=False
        )
        
        print("\n✓ Environment created successfully")
        
        # Test reset
        obs = env.reset()
        print(f"✓ Reset successful, obs shape: {obs.shape}")
        
        # Test step
        action = np.array([0.5, 0.5, 0.0, 0.3])
        obs, loss, done, info = env.step(action)
        print(f"✓ Step successful, loss: {loss:.4f}")
        
        # Test rendering
        frame = env.render()
        print(f"✓ Render successful, frame shape: {frame.shape}")
        
        env.close()
        print("\n✓ All tests passed!")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
