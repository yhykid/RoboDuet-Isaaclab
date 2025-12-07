from rsl_rl.env import VecEnv
from roboduet.envs import DuetManagerBasedRLEnv
import gymnasium as gym
import torch
from roboduet.utils.switch import global_switch

class DuetRslRlVecEnvWrapper(VecEnv):
    def __init__(self, env: DuetManagerBasedRLEnv, clip_actions: float | None = None):
        if not isinstance(env.unwrapped, DuetManagerBasedRLEnv):
            raise ValueError(
                "The environment must be inherited from DuetManagerBasedRLEnv. Environment type:"
                f" {type(env)}"
            )
        # initialize the wrapper
        self.env = env
        self.clip_actions = clip_actions

        # store information required by wrapper
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length

        # obtain dimensions of the environment 
        self.num_actions = self.unwrapped.action_manager.total_action_dim
        self.num_obs = self.unwrapped.observation_manager.group_obs_dim["policy"][0] #63
        self.num_arm_obs = self.unwrapped.observation_manager.group_obs_dim["arm_policy"][0] #20
        self.num_dog_obs = self.unwrapped.observation_manager.group_obs_dim["dog_policy"][0] #56
        self.num_arm_privileged_obs = self.unwrapped.observation_manager.group_obs_dim["arm_critic"][0] # 9
        self.num_dog_privileged_obs = self.unwrapped.observation_manager.group_obs_dim["dog_critic"][0] # 30
        self.obs_history_length = 30
        self.num_arm_obs_history = self.num_arm_obs * self.obs_history_length
        self.num_dog_obs_history = self.num_dog_obs * self.obs_history_length
        self.dog_obs_history = torch.zeros(self.num_envs, self.num_dog_obs_history, dtype=torch.float,
                                       device=self.device, requires_grad=False)
        
        self.arm_obs_history = torch.zeros(self.num_envs,self.num_arm_obs_history, dtype=torch.float,
                                       device=self.device, requires_grad=False)

        # modify the action space to the clip range
        self._modify_action_space()

        # reset at the start since the RSL-RL runner does not call reset
        self.env.reset()

    def clear_cached(self,env_ids):
        self.dog_obs_history[env_ids,:] = 0
        self.arm_obs_history[env_ids,:] = 0  
    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    """
    Properties -- Gym.Wrapper
    """

    @property
    def cfg(self) -> object:
        """Returns the configuration class instance of the environment."""
        return self.unwrapped.cfg

    @property
    def render_mode(self) -> str | None:
        """Returns the :attr:`Env` :attr:`render_mode`."""
        return self.env.render_mode

    @property
    def observation_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`observation_space`."""
        return self.env.observation_space

    @property
    def action_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`action_space`."""
        return self.env.action_space

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__

    @property
    def unwrapped(self) -> DuetManagerBasedRLEnv:
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped

    """
    Properties
    """

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """Returns the current observations of the environment."""
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()
        return obs_dict["policy"], {"observations": obs_dict}
    
    def get_arm_observations(self) -> tuple[torch.Tensor, dict]: # todo 改这个
        """Returns the current observations of the environment."""
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped.get_arm_observations()
        arm_obs_dict = obs_dict["arm_policy"]
        arm_priv_obs_dict = obs_dict["arm_critic"]
        self.arm_obs_history = torch.cat((self.arm_obs_history[:, arm_obs_dict.shape[1]:], arm_obs_dict), dim=-1)
        return {"obs": arm_obs_dict,"privileged_obs":arm_priv_obs_dict,"obs_history":self.arm_obs_history}
    
    def get_dog_observations(self) -> tuple[torch.Tensor, dict]:
        """Returns the current observations of the environment."""
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped.get_dog_observations() # not defined yet
        dog_obs_dict = obs_dict["dog_policy"]
        dog_priv_obs_dict = obs_dict["dog_critic"]
        self.dog_obs_history = torch.cat((self.dog_obs_history[:, dog_obs_dict.shape[1]:], dog_obs_dict), dim=-1)
        return {"obs": dog_obs_dict,"privileged_obs":dog_priv_obs_dict,"obs_history":self.dog_obs_history}
    
    @property
    def episode_length_buf(self) -> torch.Tensor:
        """The episode length buffer."""
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        """Set the episode length buffer.

        Note:
            This is needed to perform random initialization of episode lengths in RSL-RL.
        """
        self.unwrapped.episode_length_buf = value

    """
    Operations - MDP
    """

    def seed(self, seed: int = -1) -> int:  # noqa: D102
        return self.unwrapped.seed(seed)

    def reset(self) -> tuple[torch.Tensor, dict]:  # noqa: D102
        # reset the environment
        obs_dict, _ = self.env.reset()
        # return observations
        return obs_dict["policy"], {"observations": obs_dict}

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if not global_switch.switch_open:
            actions[:,self.unwrapped.num_actions_loco:self.unwrapped.num_actions_loco+self.unwrapped.num_actions_arm] = 0
        # clip actions
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # record step information
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        # compute dones for compatibility with RSL-RL
        dones = (terminated | truncated).to(dtype=torch.long)
        # move extra observations to the extras dict
        obs = obs_dict["policy"]
        extras["observations"] = obs_dict
        # move time out information to the extras dict
        # this is only needed for infinite horizon tasks
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        # return the step information
        return rew, rew, dones, extras # obs rew

    def close(self):  # noqa: D102
        return self.env.close()

    """
    Helper functions
    """

    def _modify_action_space(self):
        """Modifies the action space to the clip range."""
        if self.clip_actions is None:
            return

        # modify the action space to the clip range
        # note: this is only possible for the box action space. we need to change it in the future for other action spaces.
        self.env.unwrapped.single_action_space = gym.spaces.Box(
            low=-self.clip_actions, high=self.clip_actions, shape=(self.num_actions,)
        )
        self.env.unwrapped.action_space = gym.vector.utils.batch_space(
            self.env.unwrapped.single_action_space, self.num_envs
        )
    def plan(self, obs):
        rescaled_obs = obs * 0.4
        self.env.commands_dog[:, 3] = torch.clip(rescaled_obs[..., 0],0,0.01 / 4 * 3.) # [n, 2]
        self.env.commands_dog[:, 4] = torch.clip(rescaled_obs[..., 1],0,0.01 / 4 * 3.) # [n, 2]
        self.plan_actions[:] = rescaled_obs