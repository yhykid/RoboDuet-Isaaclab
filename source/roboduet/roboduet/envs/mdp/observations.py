# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""
from __future__ import annotations
import torchvision
import torch
from typing import TYPE_CHECKING
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster, RayCasterCamera
from isaaclab.assets import Articulation
from isaaclab.utils.math  import euler_xyz_from_quat, wrap_to_pi,quat_apply , quat_from_euler_xyz, quat_rotate, quat_mul
from roboduet.envs.mdp.duet_events import DuetEvent 
from collections.abc import Sequence
import numpy as np 
import cv2
from params_proto import PrefixProto, ParamsProto
from roboduet.utils.switch import global_switch
from roboduet.utils.math import *
if TYPE_CHECKING:
    from roboduet.envs import DuetManagerBasedRLEnv
    from isaaclab.managers import ObservationTermCfg

class obs_scales(PrefixProto, cli=False):
        lin_vel = 2.0
        ang_vel = 0.25
        dof_pos = 1.0
        dof_vel = 0.05
        imu = 0.1
        height_measurements = 5.0
        friction_measurements = 1.0
        body_height_cmd = 2.0
        gait_phase_cmd = 1.0
        gait_freq_cmd = 1.0
        footswing_height_cmd = 0.15
        body_pitch_cmd = 1.
        body_roll_cmd = 1.
        aux_reward_cmd = 1.0
        compliance_cmd = 1.0
        stance_width_cmd = 1.0
        stance_length_cmd = 1.0
        segmentation_image = 1.0
        rgb_image = 1.0
        depth_image = 1.0
class normalization(PrefixProto, cli=False):
        clip_observations = 100.
        clip_actions = 100.

        friction_range = [0.05, 4.5]
        ground_friction_range = [0.05, 4.5]
        restitution_range = [0, 1.0]
        added_mass_range = [-1., 3.]
        com_displacement_range = [-0.1, 0.1]
        motor_strength_range = [0.9, 1.1]
        motor_offset_range = [-0.05, 0.05]
        Kp_factor_range = [0.8, 1.3]
        Kd_factor_range = [0.5, 1.5]
        joint_friction_range = [0.0, 0.7]
        contact_force_range = [0.0, 50.0]
        contact_state_range = [0.0, 1.0]
        body_velocity_range = [-6.0, 6.0]
        foot_height_range = [0.0, 0.15]
        body_height_range = [0.0, 0.60]
        gravity_range = [-1.0, 1.0]
        motion = [-0.01, 0.01]
class RoboDuetObservations(ManagerTermBase):

    def __init__(self, cfg: ObservationTermCfg, env: DuetManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.contact_sensor: ContactSensor = env.scene.sensors['contact_forces']
        self.duet_event: DuetEvent =  env.duet_manager.get_term(cfg.params["parkour_name"])
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.sensor_cfg = cfg.params["sensor_cfg"]
        self.asset_cfg = cfg.params["asset_cfg"]
        self.env = env
        # self.body_id = self.asset.find_bodies('base')[0]
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                        requires_grad=False)
    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._obs_history_buffer[env_ids, :, :] = 0. 

    def __call__(
        self,
        env: DuetManagerBasedRLEnv,        
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        ) -> torch.Tensor:
        
        roll, pitch, yaw = euler_xyz_from_quat(self.asset.data.root_quat_w)
        
        imu_obs = torch.stack((wrap_to_pi(roll), wrap_to_pi(pitch)), dim=1).to(self.device)

        commands_dog,commands_arm = env.command_manager.get_command('base_velocity')

        obs_buf = torch.cat(((self.asset.data.joint_pos[:,:18] - self.asset.data.default_joint_pos[:,:18]) * obs_scales.dof_pos, #18
                            self.asset.data.joint_vel * obs_scales.dof_vel, # 12
                            env.action_manager.get_term('joint_pos')._leg_raw_actions, #12
                            env.action_manager.get_term('joint_pos')._arm_raw_actions, # 8 
                            commands_dog[:,:3],# todo :看看command scale怎么加
                            commands_arm if global_switch.switch_open else torch.zeros_like(commands_arm[:]),
                            roll.unsqueeze(1), #1
                            pitch.unsqueeze(1),
                            self.clock_inputs # 4 #todo 先放在这
                            ), dim=-1)

        return obs_buf 

    def _get_contact_fill(
        self,
        ):
        contact_forces = self.contact_sensor.data.net_forces_w_history[:, 0, self.sensor_cfg.body_ids] #(N, 4, 3)
        contact = torch.norm(contact_forces, dim=-1) > 2.
        previous_contact_forces = self.contact_sensor.data.net_forces_w_history[:, -1, self.sensor_cfg.body_ids] # N, 4, 3
        last_contacts = torch.norm(previous_contact_forces, dim=-1) > 2.
        contact_filt = torch.logical_or(contact, last_contacts) 
        return (contact_filt.float()-0.5).to(self.device)
    
    def _get_priv_explicit(
        self,
        ):
        base_lin_vel = self.asset.data.root_lin_vel_b 
        return torch.cat((base_lin_vel * 2.0,
                        0 * base_lin_vel,
                        0 * base_lin_vel), dim=-1).to(self.device)
    
    def _get_priv_latent(
        self,
        ):
        body_mass = self.asset.root_physx_view.get_masses()[:,self.body_id].to(self.device)
        body_com = self.asset.data.com_pos_b[:,self.body_id,:].to(self.device).squeeze(1)
        mass_params_tensor = torch.cat([body_mass, body_com],dim=-1).to(self.device)
        friction_coeffs_tensor = self.asset.root_physx_view.get_material_properties()[:, 0, 0]
        joint_stiffness = self.asset.data.joint_stiffness.to(self.device)
        default_joint_stiffness = self.asset.data.default_joint_stiffness.to(self.device)
        joint_damping = self.asset.data.joint_damping.to(self.device)
        default_joint_damping = self.asset.data.default_joint_damping.to(self.device)
        return torch.cat((
            mass_params_tensor,
            friction_coeffs_tensor.unsqueeze(1).to(self.device),
            (joint_stiffness/ default_joint_stiffness) - 1, 
            (joint_damping/ default_joint_damping) - 1
        ), dim=-1).to(self.device)
    
    def get_lpy_in_base_coord(self, env_ids):
        forward = quat_apply(self.base_quat[env_ids], self.forward_vec[env_ids])
        yaw = torch.atan2(forward[:, 1], forward[:, 0])

        self.grasper_move = torch.tensor([0.0, 0, 0.1], dtype=torch.float, device=self.device).repeat((len(env_ids), 1))
        self.grasper_move_in_world = quat_rotate(self.end_effector_state[env_ids, 3:7], self.grasper_move)
        self.grasper_in_world = self.end_effector_state[env_ids, :3] + self.grasper_move_in_world
        #print('4',self.end_effector_state[:,3:7])
        x = torch.cos(yaw) * (self.grasper_in_world[:, 0] - self.root_states[env_ids, 0]) \
            + torch.sin(yaw) * (self.grasper_in_world[:, 1] - self.root_states[env_ids, 1])
        y = -torch.sin(yaw) * (self.grasper_in_world[:, 0] - self.root_states[env_ids, 0]) \
            + torch.cos(yaw) * (self.grasper_in_world[:, 1] - self.root_states[env_ids, 1])
        z = torch.mean(self.grasper_in_world[:, 2].unsqueeze(1) - self.measured_heights, dim=1) - 0.38

        l = torch.sqrt(x**2 + y**2 + z**2)
        p = torch.atan2(z, torch.sqrt(x**2 + y**2))
        y_aw = torch.atan2(y, x)

        return torch.stack([l, p, y_aw], dim=-1)
    
    def _get_priv_obs(
        self,
        ):
        friction_coeffs_scale, friction_coeffs_shift = get_scale_shift(normalization.friction_range)
        restitutions_scale, restitutions_shift = get_scale_shift(normalization.restitution_range)
        privileged_obs_buf = torch.cat(((self.restitutions[:, 0].unsqueeze(1) - restitutions_shift) * restitutions_scale),
                                       (self.friction_coeffs[:, 0].unsqueeze(1) - friction_coeffs_shift) * friction_coeffs_scale, dim=1)

        lpy = self.get_lpy_in_base_coord(torch.arange(self.num_envs, device=self.device))
        forward = quat_apply(self.base_quat, self.forward_vec)
        yaw = torch.atan2(forward[:, 1], forward[:, 0])
        quat_base = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
        quat_ee_in_base = quat_mul(quat_base, self.end_effector_state[:, 3:7])
        privileged_obs_buf = torch.cat((privileged_obs_buf, lpy, quat_ee_in_base), dim=1)