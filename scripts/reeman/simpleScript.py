# Copyright (c)
# SPDX-License-Identifier: BSD-3-Clause

import argparse
from isaaclab.app import AppLauncher

# -------------------------------------------------------
# CLI
# -------------------------------------------------------
parser = argparse.ArgumentParser(description="Minimal custom chassis simulation in Isaac Lab / Isaac Sim.")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -------------------------------------------------------
# Imports after app launch
# -------------------------------------------------------
import torch
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# -------------------------------------------------------
# CHASSIS CONFIG (Jetbot)
# -------------------------------------------------------
JETBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Jetbot/jetbot.usd", #replace with reeman usd
    ),
    actuators={
        "wheel_acts": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            damping=None,
            stiffness=None,
        ),
    },
)

# -------------------------------------------------------
# SCENE CONFIG (Jetbot ONLY)
# -------------------------------------------------------
class ChassisSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg()
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0)
    )

    # Jetbot chassis
    Jetbot = JETBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Jetbot")

# -------------------------------------------------------
# SIMULATION LOOP
# -------------------------------------------------------
def run_simulator(sim, scene):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    while simulation_app.is_running():

        # Reset every 500 steps
        if count % 500 == 0:
            root_state = scene["Jetbot"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins

            scene["Jetbot"].write_root_pose_to_sim(root_state[:, :7])
            scene["Jetbot"].write_root_velocity_to_sim(root_state[:, 7:])

            # Reset joints
            joint_pos = scene["Jetbot"].data.default_joint_pos.clone()
            joint_vel = scene["Jetbot"].data.default_joint_vel.clone()
            scene["Jetbot"].write_joint_state_to_sim(joint_pos, joint_vel)

            scene.reset()
            print("[INFO] Resetting chassis...")

        # Control: drive forward then turn
        if count % 100 < 75:
            action = torch.Tensor([[10.0, 10.0]])
        else:
            action = torch.Tensor([[5.0, -5.0]])

        scene["Jetbot"].set_joint_velocity_target(action)

        # Step simulation
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        sim_time += sim_dt
        count += 1

# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])

    scene_cfg = ChassisSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: Scene ready!")

    run_simulator(sim, scene)

if __name__ == "__main__":
    main()
    simulation_app.close()
