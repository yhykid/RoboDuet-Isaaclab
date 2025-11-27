# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
RslRlOnPolicyRunnerCfg, 
RslRlPpoActorCriticCfg, 
RslRlPpoAlgorithmCfg,
)

################################
####### roboduet config ########
################################
    
@configclass
class DuetRslRlBaseCfg:
    num_arm_obs = 20 
    num_arm_privileged_obs = 9
    num_arm_actions = 8

    num_dog_obs = 56
    num_dog_privileged_obs = 2
    num_dog_actions = 12

    num_obs_history = 30

@configclass
class DuetArmRslRlPpoActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name = str = 'ArmActorCritic'
    num_obs: int = MISSING  
    num_privileged_obs: int = MISSING
    num_obs_history: int = MISSING
    num_actions: int = MISSING
    init_noise_std: float = MISSING
    actor_hidden_dims: list[int] = MISSING
    critic_hidden_dims: list[int] = MISSING
    activation: str = MISSING
    adaptation_module_branch_hidden_dims: list[int] = MISSING
    use_decoder: bool = MISSING
  

@configclass
class DuetDogRslRlPpoActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name = str = 'DogActorCritic'
    num_obs: int = MISSING
    num_privileged_obs: int = MISSING
    num_obs_history: int = MISSING
    num_actions: int = MISSING
    init_noise_std: float = MISSING
    actor_hidden_dims: list[int] = MISSING
    critic_hidden_dims: list[int] = MISSING
    activation: str = MISSING
    adaptation_module_branch_hidden_dims: list[int] = MISSING
    use_decoder: bool = MISSING

@configclass
class DuetRslRlPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = 'PPODuet'
    value_loss_coef: float = MISSING
    use_clipped_value_loss: bool = MISSING
    clip_param: float = MISSING
    entropy_coef: float = MISSING
    desired_kl: float = MISSING
    num_learning_epochs: int = MISSING
    num_mini_batches: int = MISSING
    learning_rate: float = MISSING
    schedule: str = MISSING
    gamma: float = MISSING
    lam: float =MISSING
    max_grad_norm: float = MISSING
    adaptation_module_learning_rate: float = MISSING
    num_adaptation_module_substeps: int = MISSING
    selective_adaptation_module_loss: bool = MISSING
    

@configclass
class DuetRslRlOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
    arm_policy: DuetArmRslRlPpoActorCriticCfg = MISSING
    dog_policy: DuetDogRslRlPpoActorCriticCfg = MISSING
    algorithm: DuetRslRlPpoAlgorithmCfg = MISSING
