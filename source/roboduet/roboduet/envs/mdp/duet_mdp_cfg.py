from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.envs.mdp.events import ( 
randomize_rigid_body_mass,
apply_external_force_torque,
reset_joints_by_scale

)
from isaaclab.envs.mdp.rewards import undesired_contacts
from isaaclab.envs.mdp.actions import JointPositionActionCfg , JointEffortActionCfg
from roboduet.envs.mdp.duet_actions import MixedPDArmMultiLegJointPositionActionCfg 
from roboduet.envs.mdp import terminations, rewards, duet_events, events, observations, duet_commands

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = duet_commands.DuetCommandCfg(
        asset_name="robot",
        resampling_time_range=(6.0,6.0 ),
        heading_control_stiffness=0.8,
        ranges=duet_commands.DuetCommandCfg.Ranges(
            lin_vel_x=(0.3, 0.8), 
            heading=(-1.6, 1.6)
        ),
        clips= duet_commands.DuetCommandCfg.Clips(
            lin_vel_clip = 0.2,
            ang_vel_clip = 0.4
        )
    )

@configclass
class DuetEventsCfg:
    """Command specifications for the MDP."""
    base_parkour = duet_events.DuetEventsCfg(
        asset_name = 'robot',
        )


@configclass
class DuetObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # observation terms (order preserved)
        roboduet_observations = ObsTerm(
            func=observations.RoboDuetObservations,
            params={            
            "asset_cfg":SceneEntityCfg("robot"),
            "sensor_cfg":SceneEntityCfg("contact_forces", body_names=".*_foot"),
            },
            clip= (-100,100)
        )
        # arm_observations = 
    policy: PolicyCfg = PolicyCfg()


@configclass
class DuetRewardsCfg:
    """Reward terms for the MDP.
    ['base', 
    'FL_hip', 
    'FL_thigh', 
    'FL_calf', 
    'FL_foot', 
    'FR_hip', 
    'FR_thigh', 
    'FR_calf', 
    'FR_foot', 
    'Head_upper', 
    'Head_lower', 
    'RL_hip', 
    'RL_thigh', 
    'RL_calf', 
    'RL_foot', 
    'RR_hip', 
    'RR_thigh', 
    'RR_calf',
    'RR_foot']
    """
# Available Body strings: 
    reward_collision = RewTerm(
        func=rewards.reward_collision, 
        weight=-10., 
        params={
            "sensor_cfg":SceneEntityCfg("contact_forces", body_names=["base",".*_calf",".*_thigh"]),
        },
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    total_terminates = DoneTerm(
        func=terminations.terminate_episode, 
        time_out=True,
        params= {
            "asset_cfg":SceneEntityCfg("robot")
        },
    )
    
@configclass
class EventCfg:
    ### Modified origin events, plz see relative issue https://github.com/isaac-sim/IsaacLab/issues/1955
    """Configuration for events."""
    reset_root_state = EventTerm(
        func= events.reset_root_state,
        params = {'offset': 3.},
        mode="reset",
    )
    reset_robot_joints = EventTerm(
        func= reset_joints_by_scale, 
        params={
            "position_range": (0.95, 1.05),
            "velocity_range": (0.0, 0.0),
        },
        mode="reset",
    )
    physics_material = EventTerm( # Okay
        func=events.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "friction_range": (0.6, 2.0),
            "num_buckets": 64,
        },
    )

    randomize_rigid_body_mass = EventTerm(
        func= randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1., 3.0),
            "operation": "add",
            },
    )
    randomize_rigid_body_com = EventTerm(
        func= events.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {'x':(-0.02, 0.02),'y':(-0.02, 0.02),'z':(-0.02, 0.02)}
            },
    )
    random_camera_position = EventTerm(
        func= events.random_camera_position,
        mode="startup",
        params={'sensor_cfg':SceneEntityCfg("depth_camera"),
                'rot_noise_range': {'pitch':(-5, 5)},
                'convention':'ros',
                },
    )
    push_by_setting_velocity = EventTerm( # Okay
        func = events.push_by_setting_velocity, 
        params={'velocity_range':{"x":(-0.5, 0.5), "y":(-0.5, 0.5)}},
        interval_range_s = (8. ,8. ),
        is_global_time= True, 
        mode="interval",
    )
    base_external_force_torque = EventTerm(  # Okay
        func=apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

@configclass
class ActionsCfg:
    joint_pos = MixedPDArmMultiLegJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        arm_joint_names=[ "zarx_j[1-8]"],
        leg_joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        scale=0.25,
    )