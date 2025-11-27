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
        dof_pos = 1.0
        dof_vel = 0.05
        lin_vel = 2.0
        ang_vel = 0.25
        body_pitch_cmd = 1.
        body_roll_cmd = 1.
class normalization(PrefixProto, cli=False):
        friction_range = [0.05, 4.5]
        restitution_range = [0, 1.0]


class RoboDuetObservations(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: DuetManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.asset_cfg = cfg.params["asset_cfg"]
        self.env = env
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        
    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.clock_inputs[env_ids, :] = 0. 

    def __call__(
        self,
        env: DuetManagerBasedRLEnv,        
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        ) -> torch.Tensor:
        
        roll, pitch, yaw = euler_xyz_from_quat(self.asset.data.root_quat_w)
        
        commands_dog,commands_arm = env.command_manager.get_command('base_velocity')

        obs_buf = torch.cat(((self.asset.data.joint_pos[:,:18] - self.asset.data.default_joint_pos[:,:18]) * obs_scales.dof_pos, #18
                            self.asset.data.joint_vel[:,:12] * obs_scales.dof_vel, # 12
                            env.action_manager.get_term('joint_pos')._leg_raw_actions, #12
                            env.action_manager.get_term('joint_pos')._arm_raw_actions[:,:6], # 6 
                            commands_dog*torch.tensor([obs_scales.lin_vel, obs_scales.lin_vel, obs_scales.ang_vel, obs_scales.body_pitch_cmd, obs_scales.body_roll_cmd], 
                                                      device=self.device, requires_grad=False, )[:,:3],
                            commands_arm if global_switch.switch_open else torch.zeros_like(commands_arm[:]),
                            roll.unsqueeze(1), #1
                            pitch.unsqueeze(1),
                            self.clock_inputs # 4 #todo 先放在这
                            ), dim=-1)
        return obs_buf


    def get_lpy_in_base_coord(self, env_ids):
        forward = quat_apply(self.base_quat[env_ids], self.forward_vec[env_ids])
        yaw = torch.atan2(forward[:, 1], forward[:, 0])

        self.grasper_move = torch.tensor([0.0, 0, 0.1], dtype=torch.float, device=self.device).repeat((len(env_ids), 1))
        self.grasper_move_in_world = quat_rotate(self.end_effector_state[env_ids, 3:7], self.grasper_move)
        self.grasper_in_world = self.end_effector_state[env_ids, :3] + self.grasper_move_in_world
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


class ArmObservations(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: DuetManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.asset_cfg = cfg.params["asset_cfg"]
        self.env = env
        self.body_id = self.asset.find_bodies('zarx_body6')[0]
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        
    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.clock_inputs[env_ids, :] = 0. 

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
        self.arm_observations = 0
        return obs_buf


    def get_lpy_in_base_coord(self, env_ids):
        forward = quat_apply(self.base_quat[env_ids], self.forward_vec[env_ids])
        yaw = torch.atan2(forward[:, 1], forward[:, 0])

        self.grasper_move = torch.tensor([0.0, 0, 0.1], dtype=torch.float, device=self.device).repeat((len(env_ids), 1))
        self.grasper_move_in_world = quat_rotate(self.end_effector_state[env_ids, 3:7], self.grasper_move)
        self.grasper_in_world = self.end_effector_state[env_ids, :3] + self.grasper_move_in_world
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
