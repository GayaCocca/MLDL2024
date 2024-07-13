"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 and Uniform Domain Randomization (https://stable-baselines3.readthedocs.io/en/master/)
"""

import gym
from env.custom_hopper_grid import *
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def main():
    # Load the best mass ranges from the tuning phase
    best_mass_ranges = np.load('best_mass_range.npy', allow_pickle=True)

    # Register the environment with the best mass ranges
    env_id = 'CustomHopperRand-v1000'
    gym.envs.register(
        id=env_id,
        entry_point="%s:CustomHopperRand" % __name__,
        max_episode_steps=500,
        kwargs={"mass_ranges": best_mass_ranges}
    )

    train_env = gym.make(env_id)

    n_policies = 3
    eval_interval = 1000 # Evaluate every 1000 episodes
    total_timesteps = 100000
    source_rewards = {i: [] for i in range(eval_interval, total_timesteps + 1, eval_interval)}
    
    test_env = gym.make('CustomHopperRand-target-v0')
    test_env = Monitor(test_env) # Wrap the evaluation environment with Monitor

    for _ in range(n_policies):
        train_env = gym.make(env_id)
        train_env = Monitor(train_env) # Wrap the training environment with Monitor
        print('first loop') 
        model = PPO('MlpPolicy', train_env, learning_rate=0.001, gamma = 0.99 , verbose=0)	

    # Evaluate the final model	
        for step in range(eval_interval, total_timesteps + 1, eval_interval):
            model.learn(total_timesteps= eval_interval, reset_num_timesteps=False)
            print('second loop')
            mean_reward, _ = evaluate_policy(model, test_env, n_eval_episodes=50, render=False)
            source_rewards[step].append(mean_reward)

    # Prepare data for plotting
    np.save('UDR_results.npy', source_rewards)
    plot_data = []
    for step in source_rewards:
        for reward in source_rewards[step]:
            plot_data.append(['Source-Target', step, reward])

    df = pd.DataFrame(plot_data, columns=['Environment', 'Timesteps', 'Mean Reward'])

    # Plot the results
    plt.figure(figsize=(12, 8))
    sns.lineplot(x='Timesteps', y='Mean Reward', hue='Environment', data=df, errorbar='sd')
    plt.title('UDR Performance')
    plt.ylabel('Mean Reward')
    plt.xlabel('Training Timesteps')
    plt.show()

if __name__ == '__main__':
    main()
