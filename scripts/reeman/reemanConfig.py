from isaaclab.app import AppLauncher
from isaaclab.utils.configclass import configclass
import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

@configclass
class ReemanRLConfig:
    """All config values for your RL environment."""
    # ----- Training Stuff -----
    max_episode_len: int = 300
    action_scale: float = 10.0           # wheel speed
    reward_forward: float = 1.0
    reward_alive: float = 0.1
    reward_turn_penalty: float = -0.001

    # ----- Robot Asset -----
    robot_cfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Reeman",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Jetbot/jetbot.usd"
        ),
        actuators={
            "wheel_vel": sim_utils.ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit_sim=None,
                damping=None,
                stiffness=None,
            )
        }
    )
