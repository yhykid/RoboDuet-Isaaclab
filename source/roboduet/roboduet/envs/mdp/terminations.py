# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math  import euler_xyz_from_quat, wrap_to_pi
from roboduet.envs.mdp import DuetEvent
from roboduet.utils.switch import global_switch

if TYPE_CHECKING:
    from roboduet.envs import DuetManagerBasedRLEnv

def terminate_episode(
    env: DuetManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):  
    reset_buf = torch.zeros((env.num_envs, ), dtype=torch.bool, device=env.device)
    asset: Articulation = env.scene[asset_cfg.name]
    roll, pitch, _ = euler_xyz_from_quat(asset.data.root_state_w[:,3:7])

    time_out_buf = env.episode_length_buf >= env.max_episode_length
    height_cutoff = asset.data.root_state_w[:, 2] < 0.28
    reset_buf |= time_out_buf
    reset_buf |= height_cutoff
    
    reverse_buf = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device, requires_grad=False)
    commands_dog,commands_arm = env.command_manager.get_command('base_velocity')
    p_align = commands_arm[:, 1]
    l_align = commands_arm[:, 0]
    delta_z = l_align*torch.sin(p_align) + 0.38 - asset.data.root_state_w[:, 2]
    if global_switch.switch_open :
        reverse_buf1 = torch.logical_and(pitch < -0.2, delta_z < -0.1) # lpy
        reverse_buf2 = torch.logical_and(pitch > 0.2, delta_z > 0.1) # lpy
        reverse_buf |= reverse_buf1 | reverse_buf2
        time_exceed_half = (env.arm_time_buf / (env.T_trajs / env.step_dt)) > 0.6
        reverse_buf = reverse_buf & time_exceed_half
        reset_buf |= reverse_buf
    
    return reset_buf
