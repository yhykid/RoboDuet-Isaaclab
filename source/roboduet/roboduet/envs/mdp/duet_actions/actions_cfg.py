from dataclasses import MISSING

from isaaclab.controllers import DifferentialIKControllerCfg, OperationalSpaceControllerCfg
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.envs.mdp.actions import JointActionCfg

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from .joint_actions import MixedPDArmMultiLegJointPositionAction


@configclass
class MixedPDArmMultiLegJointPositionActionCfg(JointActionCfg):
    """Configuration for the joint position action term.

    See :class:`JointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = (
    MixedPDArmMultiLegJointPositionAction
    )

    arm_joint_names: list[str] = MISSING
    leg_joint_names: list[str] = MISSING


    use_default_offset: bool = True
    """Whether to use default joint positions configured in the articulation asset as offset.
    Defaults to True.

    If True, this flag results in overwriting the values of :attr:`offset` to the default joint positions
    from the articulation asset.
    """
