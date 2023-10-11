import numpy as np
from train_il import BCModel
import gym_carlo
import gym
import time
import argparse
import torch
from utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--scenario",
        type=str,
        help="intersection, circularroad, lanechange",
        default="intersection",
    )
    parser.add_argument(
        "--goal",
        type=str,
        help="left, straight, right, inner, outer, all",
        default="all",
    )
    parser.add_argument("--visualize", action="store_true", default=False)
    args = parser.parse_args()
    scenario_name = args.scenario.lower()
    assert scenario_name in scenario_names, "--scenario argument is invalid!"

    if args.goal.lower() == "all":
        env = gym.make(scenario_name + "Scenario-v0", goal=len(goals[scenario_name]))
    else:
        env = gym.make(
            scenario_name + "Scenario-v0",
            goal=np.argwhere(np.array(goals[scenario_name]) == args.goal.lower())[0, 0],
        )  # hmm, unreadable

    bc_model = BCModel(obs_sizes[scenario_name], 2)
    ckpt_path = "./policies/" + scenario_name + "_" + args.goal.lower() + "_IL"
    bc_model.load_state_dict(torch.load(ckpt_path))

    episode_number = 10 if args.visualize else 100
    success_counter = 0
    env.T = 200 * env.dt - env.dt / 2.0  # Run for at most 200dt = 20 seconds
    for _ in range(episode_number):
        env.seed(int(np.random.rand() * 1e6))
        obs, done = env.reset(), False
        if args.visualize:
            env.render()
        while not done:
            t = time.time()
            obs = np.array(obs).reshape(1, -1)
            action = bc_model(torch.Tensor(obs)).detach().numpy().reshape(-1)
            obs, _, done, _ = env.step(action)
            if args.visualize:
                env.render()
                while time.time() - t < env.dt / 2:
                    pass  # runs 2x speed. This is not great, but time.sleep() causes problems with the interactive controller
            if done:
                env.close()
                if args.visualize:
                    time.sleep(1)
                if env.target_reached:
                    success_counter += 1
    if not args.visualize:
        print("Success Rate = " + str(float(success_counter) / episode_number))
