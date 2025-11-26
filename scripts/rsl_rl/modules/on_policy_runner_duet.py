import copy
import os
import os.path as osp
import shutil
import statistics
import time
from collections import deque
import cv2
import imageio
import numpy as np
import torch
from params_proto import PrefixProto
from roboduet.utils.switch import global_switch
import rsl_rl
from rsl_rl.env import VecEnv
from .ac_duet import ArmActorCritic,DogActorCritic
from rsl_rl.utils import store_code_state
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from .ppo_duet import PPODuet
from rsl_rl.modules import EmpiricalNormalization

class RunnerArgs(PrefixProto, cli=False):
    # runner
    algorithm_class_name = 'PPO'
    num_steps_per_env = 24  # per iteration
    max_iterations = 1500  # number of policy updates

    # logging
    save_interval = 400  # check for potential saves every this many iterations
    save_video_interval = 400
    log_freq = 10
    log_video = True
    
    # load and resume
    load_run = -1  # -1 = last run
    checkpoint = -1  # -1 = last saved model

class OnPolicyRunnerDuet(OnPolicyRunner):
    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"] #PPODuet
        self.arm_policy_cfg = train_cfg["arm_policy"] #arm  ac
        self.dog_policy_cfg = train_cfg["dog_policy"] #dog ac
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        self._configure_multi_gpu()

        if self.alg_cfg["class_name"] == "PPODuet":
            self.training_type = "rl"
        else:
            raise ValueError(f"Training type not found for algorithm {self.alg_cfg['class_name']}.")

        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]
        
        if self.training_type == "rl":
            if "critic" in extras["observations"]:
                self.privileged_obs_type = "critic"  # actor-critic reinforcement learnig, e.g., PPO
            else:
                self.privileged_obs_type = None
        if self.training_type == "distillation":
            if "teacher" in extras["observations"]:
                self.privileged_obs_type = "teacher"  # policy distillation
            else:
                self.privileged_obs_type = None

        if self.privileged_obs_type is not None:
            num_privileged_obs = extras["observations"][self.privileged_obs_type].shape[1]
        else:
            num_privileged_obs = num_obs

        arm_policy_class = eval(self.arm_policy_cfg.pop("class_name"))
        arm_policy: ArmActorCritic = arm_policy_class( 
                                                    #     self.arm_policy_cfg['num_obs'],
                                                    #    self.arm_policy_cfg['num_privileged_obs'],
                                                    #    self.arm_policy_cfg['num_obs_history'],
                                                    #    self.arm_policy_cfg['num_actions'], 
                                                       **self.arm_policy_cfg
                                                     ).to(self.device)
        dog_policy_class = eval(self.dog_policy_cfg.pop("class_name"))
        dog_policy: ArmActorCritic = dog_policy_class( 
                                                    #     self.dog_policy_cfg['num_obs'],
                                                    #    self.dog_policy_cfg['num_privileged_obs'],
                                                    #    self.dog_policy_cfg['num_obs_history'],
                                                    #    self.dog_policy_cfg['num_actions'], 
                                                         **self.dog_policy_cfg
                                                     ).to(self.device)

        if "rnd_cfg" in self.alg_cfg and self.alg_cfg["rnd_cfg"] is not None:
            # check if rnd gated state is present
            rnd_state = extras["observations"].get("rnd_state")
            if rnd_state is None:
                raise ValueError("Observations for the key 'rnd_state' not found in infos['observations'].")
            # get dimension of rnd gated state
            num_rnd_state = rnd_state.shape[1]
            # add rnd gated state to config
            self.alg_cfg["rnd_cfg"]["num_states"] = num_rnd_state
            # scale down the rnd weight with timestep (similar to how rewards are scaled down in legged_gym envs)
            self.alg_cfg["rnd_cfg"]["weight"] *= env.unwrapped.step_dt

        # if using symmetry then pass the environment config object
        if "symmetry_cfg" in self.alg_cfg and self.alg_cfg["symmetry_cfg"] is not None:
            # this is used by the symmetry function for handling different observation terms
            self.alg_cfg["symmetry_cfg"]["_env"] = env

        # initialize algorithm
        self.num_steps_per_env = train_cfg["num_steps_per_env"]
        alg_class = eval(self.alg_cfg.pop("class_name"))
        self.alg_arm: PPODuet = alg_class(arm_policy, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg)
        self.alg_arm.init_storage(
            self.training_type,
            self.env.num_envs,
            self.num_steps_per_env,
            [self.arm_policy_cfg['num_obs']],
            [self.arm_policy_cfg['num_privileged_obs']],
            [self.arm_policy_cfg['num_obs_history']],
            [self.arm_policy_cfg['num_actions']],
            [self.arm_policy_cfg['num_actions']]             
        ) 
        
        # dog_alg_class = eval(self.alg_cfg.pop("class_name"))
        self.alg_dog: PPODuet = alg_class(dog_policy, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg)
        self.alg_dog.init_storage(
            self.training_type,
            self.env.num_envs,
            self.num_steps_per_env,
            [self.dog_policy_cfg['num_obs']],
            [self.dog_policy_cfg['num_privileged_obs']],
            [self.dog_policy_cfg['num_obs_history']],
            [self.dog_policy_cfg['num_actions']],
            [self.dog_policy_cfg['num_actions']]
            )

        
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]

        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.privileged_obs_normalizer = EmpiricalNormalization(shape=[num_privileged_obs], until=1.0e8).to(
                self.device
            )
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
            self.privileged_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
        
        # Decide whether to disable logging
        # We only log from the process with rank 0 (main process)
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0
        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]
        #todo 这里加不加reset


    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        # initialize writer
        self.alg: PPODuet
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

        # randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # start learning
        obs_dict_arm = self.env.get_arm_observations()
        obs_arm, privileged_obs_arm, obs_history_arm = obs_dict_arm["obs"], obs_dict_arm["privileged_obs"], obs_dict_arm["obs_history"]
        obs_arm, privileged_obs_arm, obs_history_arm = obs_arm.to(self.device), privileged_obs_arm.to(self.device), obs_history_arm.to(
            self.device)
        self.alg_arm.policy.train()
        self.alg_dog.policy.train()

        # Book keeping
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        rewbuffer_eval = deque(maxlen=100)
        lenbuffer_eval = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        ep_infos = []

        # obs, extras = self.env.get_observations()
        # privileged_obs = extras["observations"].get(self.privileged_obs_type, obs)
        # obs, privileged_obs = obs.to(self.device), privileged_obs.to(self.device)
        # self.train_mode()  # switch to train mode (for dropout for example)

        # create buffers for logging extrinsic and intrinsic rewards
        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()
            # TODO: Do we need to synchronize empirical normalizers?
            #   Right now: No, because they all should converge to the same values "asymptotically".

        mean_value_loss_arm, mean_surrogate_loss_arm, mean_adaptation_module_loss_arm = 0, 0, 0
        mean_value_loss_dog, mean_surrogate_loss_dog, mean_adaptation_module_loss_dog = 0, 0, 0
        actions_arm = torch.zeros(self.env.num_envs, self.env.num_actions_arm, dtype=torch.float, device=self.device, requires_grad=False)
        # Start training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env + 1):
                    # add from duet
                    if global_switch.switch_open:
                        actions_arm = self.alg_arm.act(obs_arm[:self.env.num_envs], privileged_obs_arm[:self.env.num_envs],
                                                    obs_history_arm[:self.env.num_envs])
                        self.env.plan(actions_arm[..., -self.env.num_plan_actions:])

                    dog_obs_dict = self.env.get_dog_observations()
                    
                    if i == 0: pass
                    else:
                        # use for compute last value
                        obs_dog, privileged_obs_dog, obs_history_dog = dog_obs_dict["obs"], dog_obs_dict["privileged_obs"], dog_obs_dict["obs_history"]
                        self.alg_dog.process_env_step(rewards_dog[:self.env.num_envs], dones[:self.env.num_envs], infos)
                        if i == self.num_steps_per_env:
                            break
                    
                    actions_dog = self.alg_dog.act(dog_obs_dict["obs"], dog_obs_dict["privileged_obs"], dog_obs_dict["obs_history"])
                    ret = self.env.step(actions_dog, actions_arm[..., :-self.env.num_plan_actions])
                    rewards_dog, rewards_arm, dones, infos = ret
                    
                    if global_switch.switch_open:
                        obs_dict_arm = self.env.get_arm_observations()
                        obs_arm, privileged_obs_arm, obs_history_arm = obs_dict_arm["obs"], obs_dict_arm["privileged_obs"], obs_dict_arm["obs_history"]
                        
                        obs_arm, privileged_obs_arm, obs_history_arm, rewards_dog, rewards_arm, dones = obs_arm.to(self.device), privileged_obs_arm.to(self.device), obs_history_arm.to(self.device), rewards_dog.to(self.device), rewards_arm.to(self.device), dones.to(self.device)
                        self.alg_arm.process_env_step(rewards_arm[:self.env.num_envs], dones[:self.env.num_envs], infos)
                    
                    env_ids = dones.nonzero(as_tuple=False).flatten()
                    self.env.clear_cached(env_ids)

                    if self.log_dir is not None:
                        if 'train/episode' in infos:
                            ep_infos.append(infos['train/episode'])

                        cur_reward_sum += rewards_dog
                        cur_episode_length += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)

                        new_ids_train = new_ids[new_ids < self.env.num_envs]
                        rewbuffer.extend(cur_reward_sum[new_ids_train].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids_train].cpu().numpy().tolist())
                        cur_reward_sum[new_ids_train] = 0
                        cur_episode_length[new_ids_train] = 0


                        # -- intrinsic and extrinsic rewards
                        if self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop
                # compute returns
                if self.training_type == "rl" :
                    if global_switch.switch_open:
                        self.alg_arm.compute_returns(obs_history_arm[:self.env.num_envs], privileged_obs_arm[:self.env.num_envs])
                    self.alg_dog.compute_returns(obs_history_dog[:self.env.num_envs], privileged_obs_dog[:self.env.num_envs])
            # update policy
            if global_switch.switch_open:
                mean_value_loss_arm, mean_surrogate_loss_arm, mean_adaptation_module_loss_arm, mean_decoder_loss, mean_decoder_loss_student, mean_adaptation_module_test_loss, mean_decoder_test_loss, mean_decoder_test_loss_student = self.alg_arm.update(un_adapt=False)
            mean_value_loss_dog, mean_surrogate_loss_dog, mean_adaptation_module_loss_dog, mean_decoder_loss_dog, mean_decoder_loss_student_dog, mean_adaptation_module_test_loss_dog, mean_decoder_test_loss_dog, mean_decoder_test_loss_student_dog = self.alg_dog.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            global_switch.count += 1
            # log info
            if self.log_dir is not None and not self.disable_logs:
                # Log information
                self.log(locals())
                # Save model
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # Clear episode infos
            ep_infos.clear()
            # Save code state
            if it == start_iter and not self.disable_logs:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save the final model after training
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def save(self, path: str, infos=None):
        # -- Save model
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            'estimator_state_dict': self.alg.estimator.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        # -- Save RND model if used
        if self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        # -- Save observation normalizer if used
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["privileged_obs_norm_state_dict"] = self.privileged_obs_normalizer.state_dict()
        if self.depth_encoder_cfg is not None :
            saved_dict['depth_encoder_state_dict'] = self.alg.depth_encoder.state_dict()
            saved_dict['depth_actor_state_dict'] = self.alg.depth_actor.state_dict()
        # save model
        torch.save(saved_dict, path)

        # upload model to external logging service
        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True):
        loaded_dict = torch.load(path, weights_only=False)
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        self.alg.estimator.load_state_dict(loaded_dict['estimator_state_dict'])
        if self.alg.rnd:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
        if self.empirical_normalization:
            if resumed_training:
                self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["privileged_obs_norm_state_dict"])
            else:
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
        if self.depth_encoder_cfg is not None:
            if 'depth_encoder_state_dict' not in loaded_dict:
                warnings.warn("'depth_encoder_state_dict' key does not exist, not loading depth encoder...")
            else:
                print("Saved depth encoder detected, loading...")
                self.alg.depth_encoder.load_state_dict(loaded_dict['depth_encoder_state_dict'])
            if 'depth_actor_state_dict' in loaded_dict:
                print("Saved depth actor detected, loading...")
                self.alg.depth_actor.load_state_dict(loaded_dict['depth_actor_state_dict'])
            else:
                print("No saved depth actor, Copying actor critic actor to depth actor...")
                self.alg.depth_actor.load_state_dict(self.alg.policy.actor.state_dict())

        if load_optimizer and resumed_training:
            # -- algorithm optimizer
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            # -- RND optimizer if used
            if self.alg.rnd:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        # -- load current learning iteration
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_estimator_inference_policy(self, device=None):
        self.alg: PPOWithExtractor
        self.alg.estimator.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.estimator.to(device)
        return self.alg.estimator
    
    def get_depth_encoder_inference_policy(self, device=None):
        self.alg.depth_encoder.eval()
        if device is not None:
            self.alg.depth_encoder.to(device)
        return self.alg.depth_encoder

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.policy.to(device)
        policy = self.alg.policy.act_inference
        if self.cfg["empirical_normalization"]:
            if device is not None:
                self.obs_normalizer.to(device)
            policy = lambda x: self.alg.policy.act_inference(self.obs_normalizer(x))  # noqa: E731
        return policy

    def get_inference_depth_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.depth_actor.to(device)
        policy = self.alg.depth_actor
        if self.cfg["empirical_normalization"]:
            if device is not None:
                self.obs_normalizer.to(device)
            policy = lambda x: self.alg.depth_actor(self.obs_normalizer(x))  # noqa: E731
        return policy
