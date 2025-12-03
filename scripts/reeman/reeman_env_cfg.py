# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse

from isaaclab.app import AppLauncher
from isaaclab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass
from isaaclab.managers import (
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    TerminationTermCfg as DoneTerm,
    EventTermCfg as EventTerm,
)

import math
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ActionTermCfg as ActTerm

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates adding a custom robot to an Isaac Lab environment."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

#ROBOT_USD = "/home/dhruv/Downloads/purple_assy_urdf_test7_2/purple_assy_urdf_test7/urdf/purple_new_backup/purple_new_backup.usd"    
#/home/dhruv/Downloads/interbotix_ros_manipulators-humble/interbotix_ros_xsarms/interbotix_xsarm_descriptions/urdf/mobile_wx250s.usd #try to use regular usd and not backup 

#Functions: 

def has_collision(robot, env_id=0, threshold=1e-3):
    contact_forces = robot.root_physx_view.get_contact_force_tensor()
    # extract forces for this env only
    env_forces = contact_forces[env_id]
    total_force = torch.norm(env_forces, dim=-1).sum()
    return total_force > threshold

@configclass
class NewRobotsSceneCfg(InteractiveSceneCfg):
    """Designs the scene."""

    REEMAN_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path="/home/dhruv/Downloads/purple_assy_urdf_test7_2/purple_assy_urdf_test7/urdf/purple_new_backup/purple_new_backup.usd"),
    actuators={"wheel_acts": ImplicitActuatorCfg(
    joint_names_expr=["left_wheel_joint", "right_wheel_joint"],
    damping=None,
    stiffness=None
)}
)
    # Ground-plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    Reeman = REEMAN_CONFIG.replace(prim_path="/Reeman")


@configclass
class EventCfg:
    reset_base = EventTerm(
        func=lambda robot: robot.reset_root_state(),  # Placeholder reset function
        mode="reset",
        params={
            "pose_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "yaw": (-math.pi, math.pi)},
            "velocity_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.1, 0.1)},
        },
    )

@configclass
class ActionsCfg:
    left_wheel = ActTerm(low=-1.0, high=1.0)
    right_wheel = ActTerm(low=-1.0, high=1.0)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        left_wheel_vel = ObsTerm(func=lambda robot: robot.get_joint_velocity("left_wheel_joint"))
        right_wheel_vel = ObsTerm(func=lambda robot: robot.get_joint_velocity("right_wheel_joint"))
        position = ObsTerm(func=lambda robot: robot.get_root_position())
    policy: PolicyCfg = PolicyCfg()

@configclass
class RewardsCfg:
    forward_progress = RewTerm(
        func=lambda robot: robot.get_root_position()[0],
        weight=1.0,
    )

    collision_penalty = RewTerm(
        func=lambda robot: float(has_collision(robot)), 
        weight=-5.0,
    )

@configclass
class TerminationsCfg:
    collision = DoneTerm(
    func=lambda robot, threshold: has_collision(robot, threshold=threshold),
    params={"threshold": 1e-3},
)

    collision = DoneTerm(
        func=lambda robot: has_collision(robot),
        params={"threshold": 1e-3},
    )

@configclass
class TwoWheelEnvCfg(ManagerBasedRLEnvCfg):
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    scene: NewRobotsSceneCfg = NewRobotsSceneCfg()  
    policy: PolicyCfg = PolicyCfg()
    episode_length_s: float = 20.0

