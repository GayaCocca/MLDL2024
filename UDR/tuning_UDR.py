import gym
from env.custom_hopper_grid import *
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import numpy as np
import itertools

def train_and_evaluate(env_id, model_params, train_steps=10000, eval_episodes=50):
    """
    Train a PPO model on the given environment and evaluate its performance.

    Parameters:
    env_id (str): The ID of the environment to use.
    model_params (dict): Parameters to use for the PPO model.
    train_steps (int): Number of training timesteps.
    eval_episodes (int): Number of evaluation episodes.

    Returns:
    float: Mean reward over the evaluation episodes.
    """
    # Create the environment
    env = gym.make(env_id)
    # Monitor to track the training process
    env = Monitor(env)
    # Create the model
    model = PPO('MlpPolicy', env, **model_params)
    # Train the model
    model.learn(total_timesteps=train_steps)
    # Evaluate the model
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=eval_episodes, render=False)
    return mean_reward

def main():
    # Define different uniform distribution ranges for each parameter
    mass_ranges_param_1 = [(2.92699082, 5.92699082), (3.42699082, 6.42699082), (2.92699082, 5.42699082), (3.42699082, 5.42699082), (2.92699082, 5.02699082), (3.02699082, 6.02699082), (3.32699082, 6.326990829), (2.92699082, 4.92699082), (3.12699082, 6.12699082), (3.52699082, 6.52699082), (3.93, 5.93), (1.93, 5.93), (3.26, 4.6)]
    mass_ranges_param_2 = [(1.71433605, 4.71433605), (2.21433605, 5.21433605), (1.71433605, 4.21433605), (2.21433605, 4.21433605), (1.71433605, 3.81433605), (1.81433605, 4.81433605), (2.11433605, 5.11433605), (1.71433605, 3.71433605), (1.91433605, 4.91433605), (2.31433605, 5.31433605), (2.71, 4.71), (0.71, 4.71), (2.09, 3.33)]
    mass_ranges_param_3 = [(4.0893801, 7.0893801), (4.5893801, 7.5893801), (4.0893801, 6.5893801), (4.5893801, 6.5893801), (4.0893801, 6.0893801), (4.1893801, 7.1893801), (4.4893801, 7.4893801), (4.0893801, 6.0893801), (4.2893801, 7.2893801), (4.6893801, 7.6893801), (5.09, 7.09), (3.09, 7.09), (4.36, 5.82)]
    

    model_params = {
        'learning_rate': 0.001,
        'gamma': 0.99,
        'verbose': 0
    }

    best_mean_reward = -np.inf
    best_mass_range = None
    i = 1

    # Generate all combinations of the distributions
    mass_ranges_combinations = itertools.product(mass_ranges_param_1, mass_ranges_param_2, mass_ranges_param_3)

    # Tune over different mass ranges
    for mass_ranges in mass_ranges_combinations:
        env_id = f'CustomHopperRand-source-v{i}'
        # Update the environment with new mass ranges
        gym.envs.register(
            id=env_id,
            entry_point="%s:CustomHopperRand" % __name__,
            max_episode_steps=500,
            kwargs={"domain": "source", "mass_ranges": mass_ranges}
        )

        # Train and evaluate
        mean_reward = train_and_evaluate(env_id, model_params)
        print(f'Mass ranges: {mass_ranges}, Mean reward: {mean_reward}')

        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            best_mass_range = mass_ranges

        i += 1
    print(f'Best mass range: {best_mass_range}, Best mean reward: {best_mean_reward}')

    # Save the best mass range for final training
    np.save('best_mass_range.npy', best_mass_range)

if __name__ == '__main__':
    main()
