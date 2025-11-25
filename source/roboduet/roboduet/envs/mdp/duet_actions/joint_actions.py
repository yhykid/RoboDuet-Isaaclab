
from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log
from isaaclab.envs.mdp.actions import JointAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .actions_cfg import MixedPDArmMultiLegJointPositionActionCfg

class MixedPDArmMultiLegJointPositionAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    cfg: MixedPDArmMultiLegJointPositionActionCfg
    """The configuration of the action term."""

    def __init__(
        self,
        cfg: MixedPDArmMultiLegJointPositionActionCfg,
        env: ManagerBasedEnv,
    ):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint positions as offset
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[
                :, self._joint_ids
            ].clone()

        # setup the arm command buffer
        self._arm_joint_ids, self._arm_joint_names = self._asset.find_joints(
            self.cfg.arm_joint_names
        )

        self._leg_joint_ids, self._leg_joint_names = self._asset.find_joints(
            self.cfg.leg_joint_names
        )
        self._arm_raw_actions = torch.zeros(
            self.num_envs, len(self._arm_joint_ids), device=self.device
        )
        self._arm_processed_actions = torch.zeros_like(self.arm_raw_actions)

        self._leg_raw_actions = torch.zeros(
            self.num_envs, len(self._leg_joint_ids), device=self.device
        )
        self._leg_processed_actions = torch.zeros_like(self._leg_raw_actions)


    def apply_actions(self):
        """Apply the actions."""
        self._asset.set_joint_effort_target(
            self._leg_processed_actions, joint_ids=self._leg_joint_ids
        )
        self._asset.set_joint_position_target(
            self.arm_processed_actions, joint_ids=self._arm_joint_ids
        )

    def process_actions(self, actions: torch.Tensor):
        """Process the actions."""
        #todo 
        # store the raw actions
        self._raw_actions[:] = actions
        # apply the affine transformations
        self._processed_actions = self._raw_actions * self._scale + self._offset
        # store the non-command leg actions

        # store the raw arm actions, which is the target joint pos
        self._arm_raw_actions[:] = self.command.arm_joint_sub_goal
        self._arm_processed_actions[:] = self._arm_raw_actions.clone()

        self._leg_processed_actions[:] = self._leg_raw_actions.clone()

    @property
    def arm_raw_actions(self) -> torch.Tensor:
        """Get the raw arm actions."""
        return self._arm_raw_actions

    @property
    def arm_processed_actions(self) -> torch.Tensor:
        """Get the processed arm actions."""
        return self._arm_processed_actions

    @property
    def leg_raw_actions(self) -> torch.Tensor:
        """Get the raw leg actions."""
        return self._leg_raw_actions

    @property
    def leg_processed_actions(self) -> torch.Tensor:
        """Get the processed leg actions."""
        return self._leg_processed_actions
