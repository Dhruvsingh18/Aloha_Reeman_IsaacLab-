from isaaclab.app import AppLauncher
import argparse
from stable_baselines3 import PPO

from reeman_env import ReemanEnv

# -------- Isaac app launcher args --------
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch Omniverse Isaac app
app = AppLauncher(args).app


def main():
    env = ReemanEnv()

    #Create Mutiple PPO's and switch case here
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./reeman_logs",
    )

    #Soft critic, etc 

    model.learn(total_timesteps=300_000)
    model.save("reeman_policy")
    print("Saved policy!")


if __name__ == "__main__":
    main()
    app.close()

'''
isaaclab/source/standalone/workflows/reeman/
    ├── reeman_config.py
    ├── reeman_env.py
    └── train_reeman.py

     run: ./isaaclab.sh -p source/standalone/workflows/reeman/train_reeman.py


'''
