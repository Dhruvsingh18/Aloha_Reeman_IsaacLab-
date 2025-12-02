import gym
import numpy as np
import torch

from isaaclab.sim import SimulationContext
from isaaclab.assets import Articulation, SurfaceGripper, SurfaceGripperCfg
import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prim_utils

from isaaclab_assets import PICK_AND_PLACE_CFG


class PickPlaceEnv(gym.Env):

    def __init__(self, device="cpu"):
        super().__init__()

        # ---- Simulator ----
        self.sim_cfg = sim_utils.SimulationCfg(device=device)
        self.sim = SimulationContext(self.sim_cfg)

        # ---- Scene ----
        self.robot, self.gripper, self.object_prim, self.origins = self._design_scene()

        # ---- Initialize robot control ----
        self.robot.initialize()
        self.robot.set_control_mode("position")

        # ---- Spaces ----
        action_dim = self.robot.num_dof + 1
        obs_dim = self.robot.num_dof * 2 + 1  # jpos + jvel + gripper

        self.action_space = gym.spaces.Box(
            low=-1, high=1,
            shape=(action_dim,),
            dtype=np.float32
        )

        self.observation_space = gym.spaces.Box(
            low=-5, high=5,
            shape=(obs_dim,),
            dtype=np.float32
        )

        self.sim.reset()
        self.sim.play()

    # ---------------------------------------------------------

    def _design_scene(self):

        prim_utils.create_prim("/World/Origin", "Xform", [0, 0, 0])

        # robot config
        robot_cfg = PICK_AND_PLACE_CFG.copy()
        robot_cfg.prim_path = "/World/Origin/Robot"
        robot = Articulation(cfg=robot_cfg)

        # gripper
        grip_cfg = SurfaceGripperCfg()
        grip_cfg.prim_path = "/World/Origin/Robot/picker_head/SurfaceGripper"
        gripper = SurfaceGripper(cfg=grip_cfg)

        # object to pick
        cube_cfg = sim_utils.CuboidCfg(size=(0.05, 0.05, 0.05), color=(0.2, 0.2, 1.0))
        obj = cube_cfg.func("/World/Object", translation=[0.4, 0, 0.03])

        origins = torch.tensor([[0, 0, 0]], device=self.sim.device)

        return robot, gripper, obj, origins

    # ---------------------------------------------------------

    def reset(self):
        self.sim.reset()

        # reset robot
        root = self.robot.data.default_root_state.clone()
        root[:, :3] += self.origins
        self.robot.write_root_pose_to_sim(root[:, :7])
        self.robot.write_root_velocity_to_sim(root[:, 7:])

        # randomize joints
        jpos = self.robot.data.default_joint_pos.clone()
        jpos += torch.randn_like(jpos) * 0.05
        jvel = torch.zeros_like(jpos)
        self.robot.write_joint_state_to_sim(jpos, jvel)

        self.robot.reset()
        self.gripper.reset()

        # reset cube
        self.object_prim.GetAttribute("xformOp:translate").Set([0.4, 0, 0.03])

        return self._get_obs()

    # ---------------------------------------------------------

    def step(self, action):

        arm_action = action[:-1]
        grip_cmd = action[-1]

        # apply arm
        arm_tensor = torch.tensor(arm_action, device=self.sim.device)
        self.robot.set_joint_positions(arm_tensor)
        self.robot.write_data_to_sim()

        # apply gripper
        grip_tensor = torch.tensor([grip_cmd], device=self.sim.device)
        self.gripper.set_grippers_command(grip_tensor)
        self.gripper.write_data_to_sim()

        self.sim.step()
        self.gripper.update(self.sim.get_physics_dt())

        obs = self._get_obs()
        reward = self._compute_reward()
        done = False

        return obs, reward, done, {}

    # ---------------------------------------------------------

    def _get_obs(self):
        jpos = self.robot.get_joint_positions().cpu().numpy().flatten()
        jvel = self.robot.get_joint_velocities().cpu().numpy().flatten()
        grip = self.gripper.state.cpu().numpy()

        return np.concatenate([jpos, jvel, grip])

    # ---------------------------------------------------------

    def _compute_reward(self):
        return 0.0
