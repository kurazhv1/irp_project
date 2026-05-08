"""
Port of MujocoCompensatedPDController to MuJoCo 2.3.7+ API

Original uses mujoco-py (old API), this uses mujoco (new API)
"""

import numpy as np
import mujoco


class MujocoCompensatedPDController:
    """
    Compensated joint-space PD controller for MuJoCo 2.3.7+
    
    Compensates for gravity, Coriolis, and centrifugal forces.
    """
    
    def __init__(self, model, data, joint_names, kp=1, kv=1):
        self.model = model
        self.data = data
        self.kp = kp
        self.kv = kv
        
        # Get joint IDs
        self.joint_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in joint_names
        ]
        
        # Get joint position and velocity addresses
        self.joint_pos_addrs = []
        self.joint_vel_addrs = []
        
        for joint_id in self.joint_ids:
            # For simple joints (hinge/slide), qpos and qvel indices are direct
            qpos_addr = model.jnt_qposadr[joint_id]
            qvel_addr = model.jnt_dofadr[joint_id]
            
            self.joint_pos_addrs.append(qpos_addr)
            self.joint_vel_addrs.append(qvel_addr)
        
        self.joint_pos_addrs = np.array(self.joint_pos_addrs)
        self.joint_vel_addrs = np.array(self.joint_vel_addrs)
        
        self.N_JOINTS = len(self.joint_ids)
        
    @property
    def q(self):
        """Current joint positions"""
        return self.data.qpos[self.joint_pos_addrs].copy()
    
    @property
    def dq(self):
        """Current joint velocities"""
        return self.data.qvel[self.joint_vel_addrs].copy()
    
    def M(self):
        """Get mass/inertia matrix"""
        # MuJoCo stores M in data.qM (sparse format)
        # We need to extract submatrix for our joints
        M_full = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, M_full, self.data.qM)
        
        # Extract submatrix for controlled joints
        M_sub = M_full[np.ix_(self.joint_vel_addrs, self.joint_vel_addrs)]
        return M_sub
    
    def g(self):
        """Get gravity/bias forces (qfrc_bias)
        
        qfrc_bias contains effects of Coriolis, centrifugal, and gravitational forces
        """
        return self.data.qfrc_bias[self.joint_vel_addrs].copy()
    
    def generate(self, target, target_velocity=None):
        """
        Generate control signal with dynamics compensation
        
        Args:
            target: target joint positions
            target_velocity: target joint velocities (default: zeros)
            
        Returns:
            u: control torques
        """
        if target_velocity is None:
            target_velocity = np.zeros(self.N_JOINTS)
        
        q = self.q
        dq = self.dq
        
        # Position and velocity errors
        q_tilde = target - q
        dq_tilde = target_velocity - dq
        
        # Get dynamics matrices
        M = self.M()
        g = self.g()
        
        # Compensated PD control:
        # u = M * (kp * q_error + kv * dq_error) - g
        u = np.dot(M, self.kp * q_tilde + self.kv * dq_tilde) - g
        
        return u
    
    def send_forces(self, u):
        """Apply control torques"""
        # Assuming first N actuators correspond to our joints
        self.data.ctrl[:self.N_JOINTS] = u
    
    def _load_state(self, qpos_init, qvel_init=None):
        """Load state (for reset)"""
        if qvel_init is None:
            qvel_init = np.zeros(self.model.nv)
        
        # Save old state
        old_qpos = self.data.qpos.copy()
        old_qvel = self.data.qvel.copy()
        old_ctrl = self.data.ctrl.copy()
        
        # Set new state
        self.data.qpos[:] = qpos_init
        self.data.qvel[:] = qvel_init
        
        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)
        
        return old_qpos, old_qvel, old_ctrl
