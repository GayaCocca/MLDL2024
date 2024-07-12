"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import gym
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import itertools

def make_env():
    return CustomHopper(domain='source') #in this way I define an istance of the class including also the domain randomization

def train_and_evaluate(learning_rate, gamma, total_timesteps=100000, n_eval_episodes=50):
    # Create the environment
    
    train_env = gym.make('CustomHopper-source-v0')
    eval_env = Monitor(gym.make('CustomHopper-source-v0'))

    # Initialize the model
    model = PPO('MlpPolicy', train_env, learning_rate=learning_rate, gamma = gamma , verbose=0)

    # Train the model
    model.learn(total_timesteps=total_timesteps)

    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=n_eval_episodes)
    
    return mean_reward, std_reward


def main():

    #TUNING PHASE 
    results = []
    learning_rates = [1e-2, 5e-3, 1e-3, 5e-4]
    gammas = [0.99, 0.999]
    param_grid = list(itertools.product(learning_rates, gammas))


    for lr, gamma in param_grid:
      print(f"Training with learning_rate={lr} and gamma={gamma}")
      mean_reward, std_reward = train_and_evaluate(lr, gamma)
      results.append((lr, gamma, mean_reward, std_reward))

     #Find the best hyperparameters based on mean reward
    best_params = max(results, key=lambda x: x[2])
    print(f"Best parameters: learning_rate={best_params[0]}, gamma={best_params[1]} with mean reward={best_params[2]} +/- {best_params[3]}")

    

if __name__ == '__main__':
    main()