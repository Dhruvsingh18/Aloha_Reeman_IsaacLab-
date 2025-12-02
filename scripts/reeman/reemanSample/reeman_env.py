import numpy as np
import gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg

from reeman_config import ReemanRLConfig


class ReemanSceneCfg(InteractiveSceneCfg):
    """Simple scene: ground + lights + chassis."""
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg()
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0)
    )

    # Insert the robot config
    reeman = ReemanRLConfig.robot_cfg


class ReemanEnv(gym.Env):
    """Isaac Lab RL environment around only the Reeman chassis."""

    def __init__(self, cfg=ReemanRLConfig()):
        super().__init__()
        self.cfg = cfg

        # ---------- Simulation Context ----------
        sim_cfg = sim_utils.SimulationCfg(device="cuda:0")
        self.sim = sim_utils.SimulationContext(sim_cfg)
        self.sim.set_camera_view([4, 0, 3], [0, 0, 0])

        # ---------- Scene ----------
        scene_cfg = ReemanSceneCfg(num_envs=1, env_spacing=2.0)
        self.scene = InteractiveScene(scene_cfg)
        self.sim.reset()
        self.scene.reset()

        # ---------- Observations / Actions ----------
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(6,), dtype=np.float32
        )

        # Two wheels = two actions
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32
        )

        self.step_count = 0

    # ==========================================================
    # RESET
    # ==========================================================
    def reset(self):
        self.step_count = 0
        self.scene.reset()

        state = np.zeros(6, dtype=np.float32)
        return state

    # ==========================================================
    # STEP
    # ==========================================================
    def step(self, action):
        self.step_count += 1

        # Scale wheel velocities
        left, right = action * self.cfg.action_scale
        wheel_target = torch.tensor([[left, right]], dtype=torch.float32)
        self.scene["reeman"].set_joint_velocity_target(wheel_target)

        # Step simulation
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(self.sim.get_physics_dt())

        # Fake observation for now (you can add lidar/camera/etc)
        obs = np.random.randn(6).astype(np.float32)

        # Reward: move forward
        reward = (
            self.cfg.reward_forward * ((left + right) / 2.0)
            + self.cfg.reward_alive
            + self.cfg.reward_turn_penalty * (abs(left - right))
        )

        done = self.step_count > self.cfg.max_episode_len
        info = {}

        return obs, reward, done, info
