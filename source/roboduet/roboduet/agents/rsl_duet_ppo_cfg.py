from roboduet.agents.duet_rl_cfg import (
ParkourRslRlOnPolicyRunnerCfg,
ParkourRslRlPpoActorCriticCfg,
ParkourRslRlActorCfg,
ParkourRslRlStateHistEncoderCfg,
ParkourRslRlEstimatorCfg,
ParkourRslRlPpoAlgorithmCfg,

DuetArmRslRlPpoActorCriticCfg,
DuetDogRslRlPpoActorCriticCfg,
DuetRslRlPpoAlgorithmCfg,
DuetRslRlOnPolicyRunnerCfg,

)
from isaaclab.utils import configclass

@configclass
class DuetGo2PPORunnerCfg(DuetRslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = "unitree_go2_duet"
    empirical_normalization = False
    dog_policy = DuetDogRslRlPpoActorCriticCfg(
        num_obs = 56,
        num_privileged_obs = 2,
        num_obs_history = 1680,
        num_actions = 12
    )

    arm_policy = DuetArmRslRlPpoActorCriticCfg(
        num_obs=20,
        num_privileged_obs=9,
        num_obs_history=600,
        num_actions=8,
    )
    
    
    algorithm = DuetRslRlPpoAlgorithmCfg(
        # value_loss_coef=1.0,
        # use_clipped_value_loss=True,
        # clip_param=0.2,
        # entropy_coef=0.01,
        # desired_kl=0.01,
        # num_learning_epochs=5,
        # num_mini_batches=4,
        # learning_rate = 2.e-4,
        # schedule="adaptive",
        # gamma=0.99,
        # lam=0.95,
        # max_grad_norm=1.0,
        # dagger_update_freq = 20,
        # priv_reg_coef_schedual = [0.0, 0.1, 2000.0, 3000.0],
    )

