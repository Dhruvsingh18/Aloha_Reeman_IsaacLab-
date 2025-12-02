# -------- Isaac Lab App Launcher --------
from isaaclab.app import AppLauncher
import argparse

# Add Isaac launcher arguments
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch Isaac App (spawns physics + renderer)
app = AppLauncher(args).app

# -------- RL Training Code --------
from stable_baselines3 import PPO
from reeman_pickplace_env import PickPlaceEnv

def main():
    # create your Isaac Lab environment
    env = PickPlaceEnv(device="cpu")

    # create PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./pickplace_logs",
    )

    # train
    model.learn(total_timesteps=500_000)

    # save trained policy
    model.save("pick_place_policy")
    print("Saved policy!")

if __name__ == "__main__":
    main()
    app.close()
