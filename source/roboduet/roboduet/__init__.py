# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python module serving as a project/extension template.
"""

# Register Gym environments.
# from .tasks import *

# Register UI extensions.
from .ui_extension_example import *

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Template-Roboduet-v0",
    entry_point="roboduet.envs:DuetManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.envs.duet_env_cfg:DuetGo2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_duet_ppo_cfg:DuetGo2PPORunnerCfg",
    },
)

