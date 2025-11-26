from roboduet.agents.duet_rl_cfg import (
DuetArmRslRlPpoActorCriticCfg,
DuetDogRslRlPpoActorCriticCfg,
DuetRslRlPpoAlgorithmCfg,
DuetRslRlOnPolicyRunnerCfg,
)
from isaaclab.utils import configclass
from params_proto import PrefixProto

class PPO_Args(PrefixProto):
    # algorithm
    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.01
    num_learning_epochs = 5
    num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
    learning_rate = 5.e-4  # 5.e-4
    adaptation_module_learning_rate = 5.e-4
    num_adaptation_module_substeps = 1
    schedule = 'adaptive'  # could be adaptive, fixed
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.

    selective_adaptation_module_loss = False
class ArmAC_Args(PrefixProto, cli=False):
    # policy
    init_noise_std = 0.1
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
    adaptation_module_branch_hidden_dims = [256, 128]
    use_decoder = False

class DogAC_Args(PrefixProto, cli=False):
    # policy
    init_noise_std = 1.0
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
    adaptation_module_branch_hidden_dims = [256, 128]
    use_decoder = False

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
        num_actions = 12,
        init_noise_std = DogAC_Args.init_noise_std,
        actor_hidden_dims = DogAC_Args.actor_hidden_dims,
        critic_hidden_dims = DogAC_Args.critic_hidden_dims,
        activation =  DogAC_Args.activation,
        adaptation_module_branch_hidden_dims = DogAC_Args.adaptation_module_branch_hidden_dims,
        use_decoder = DogAC_Args.use_decoder,
    )

    arm_policy = DuetArmRslRlPpoActorCriticCfg(
        num_obs=20,
        num_privileged_obs=9,
        num_obs_history=600,
        num_actions=8,
        init_noise_std=ArmAC_Args.init_noise_std,
        actor_hidden_dims=ArmAC_Args.actor_hidden_dims,
        critic_hidden_dims=ArmAC_Args.critic_hidden_dims,
        activation=ArmAC_Args.activation,
        adaptation_module_branch_hidden_dims=ArmAC_Args.adaptation_module_branch_hidden_dims,
        use_decoder=ArmAC_Args.use_decoder,
    )
    
    
    algorithm = DuetRslRlPpoAlgorithmCfg(
        value_loss_coef=PPO_Args.value_loss_coef,
        use_clipped_value_loss=PPO_Args.use_clipped_value_loss,
        clip_param=PPO_Args.clip_param,
        entropy_coef=PPO_Args.entropy_coef,
        desired_kl=PPO_Args.desired_kl,
        num_learning_epochs=PPO_Args.num_learning_epochs,
        num_mini_batches=PPO_Args.num_mini_batches,
        learning_rate = PPO_Args.learning_rate,
        schedule= PPO_Args.schedule,
        gamma=PPO_Args.gamma,
        lam=PPO_Args.lam,
        max_grad_norm=PPO_Args.max_grad_norm,
        adaptation_module_learning_rate=PPO_Args.adaptation_module_learning_rate,
        num_adaptation_module_substeps=PPO_Args.num_adaptation_module_substeps,
        selective_adaptation_module_loss=PPO_Args.selective_adaptation_module_loss,
    )

