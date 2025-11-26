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
#########################
# Policy configurations #
#########################

@configclass
class ParkourRslRlBaseCfg:
    num_priv_explicit: int = 3 + 3 + 3 # 9
    num_priv_latent: int = 4 + 1 + 12 +12 # 29
    num_prop: int = 3 + 2 + 3 + 4 + 36 + 5 # 53
    num_scan: int = 132
    num_hist: int = 10
    
@configclass
class ParkourRslRlStateHistEncoderCfg(ParkourRslRlBaseCfg):
    class_name: str = "StateHistoryEncoder" 
    channel_size: int = 10 
    
@configclass
class ParkourRslRlDepthEncoderCfg(ParkourRslRlBaseCfg):
    backbone_class_name: str = "DepthOnlyFCBackbone58x87" 
    encoder_class_name: str = "RecurrentDepthBackbone" 
    depth_shape: tuple[int] = (87, 58)
    hidden_dims: int = 512
    learning_rate: float = 1.e-3
    num_steps_per_env: int = 24 * 5

@configclass
class ParkourRslRlEstimatorCfg(ParkourRslRlBaseCfg):
    class_name: str = "DefaultEstimator" 
    train_with_estimated_states: bool = True 
    learning_rate: float = 1.e-4 
    hidden_dims: list[int] = MISSING 
    
@configclass
class ParkourRslRlActorCfg(ParkourRslRlBaseCfg):
    class_name: str = "Actor"
    state_history_encoder: ParkourRslRlStateHistEncoderCfg = MISSING


@configclass
class ParkourRslRlPpoActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = 'ActorCriticRMA'
    tanh_encoder_output: bool = False 
    scan_encoder_dims: list[int] = MISSING
    priv_encoder_dims: list[int] = MISSING
    actor: ParkourRslRlActorCfg = MISSING

@configclass
class ParkourRslRlPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = 'PPOWithExtractor'
    dagger_update_freq: int = 1
    priv_reg_coef_schedual: list[float]= [0, 0.1, 2000, 3000]

@configclass
class ParkourRslRlDistillationAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "DistillationWithExtractor"

@configclass
class ParkourRslRlOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
    policy: ParkourRslRlPpoActorCriticCfg = MISSING
    estimator: ParkourRslRlEstimatorCfg = MISSING
    depth_encoder: ParkourRslRlDepthEncoderCfg | None = None
    algorithm: ParkourRslRlPpoAlgorithmCfg | ParkourRslRlDistillationAlgorithmCfg = MISSING

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
