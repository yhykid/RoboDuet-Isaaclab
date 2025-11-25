# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass

from isaaclab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg
from roboduet.envs.roboduet_ui import DuetManagerBasedRLEnvWindow

@configclass
class DuetManagerBasedRLEnvCfg(ManagerBasedRLEnvCfg):
    ui_window_class_type: type | None = DuetManagerBasedRLEnvWindow
    roboduet: object = MISSING