"""Train an RL agent on the OpenAI Gym Hopper environment using
    REINFORCE and Actor-critic algorithms
"""
import argparse
import itertools

import matplotlib.pyplot as plt
import time
import torch
import gym
import numpy as np

from env.custom_hopper import *
from agent_TUNING import Agent, Policy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=10000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=200, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--baseline', action='store_true', help='Use baseline in REINFORCE algorithm')
    parser.add_argument('--algorithm', default='Reinforce', choices=['Reinforce', 'ActorCritic'], help='Algorithm to use: Reinforce or ActorCritic')

    return parser.parse_args()

args = parse_args()


def train(lr, gamma, algorithm, baseline, n_episodes):
    env = gym.make('CustomHopper-source-v0')
    # env = gym.make('CustomHopper-target-v0')
    
    """
		Training
	"""
    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]
    
    policy = Policy(observation_space_dim, action_space_dim)
    agent = Agent(policy, gamma=gamma, device=args.device)
    agent.set_optimizer(torch.optim.Adam(policy.parameters(), lr=lr))
    
    episode_rewards = []
    
    for episode in range(n_episodes):
        
        done = False
        train_reward = 0
        
        
        state = env.reset()  # Reset the environment and observe the initial state
        
        while not done:  # Loop until the episode is over
            action, action_probabilities = agent.get_action(state)
            previous_state = state
            
            state, reward, done, info = env.step(action.detach().cpu().numpy())
            
            agent.store_outcome(previous_state, state, action_probabilities, reward, done)
            
            train_reward += reward
            
        
        episode_rewards.append(train_reward)
        
        agent.update_policy(algorithm=algorithm, baseline=baseline)
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    
    return mean_reward, std_reward, agent
  


def main():
    
    learning_rates = [5e-3, 1e-3, 5e-4]
    gamma_values = [0.95, 0.99, 0.999]
    algorithms = ['Reinforce', 'ActorCritic']
    baselines = [True, False] # with this line it includes both baseline and non-baseline
    results = [] 
    
    param_grid = list(itertools.product(learning_rates, gamma_values, algorithms, baselines))
    
    for lr, gamma, algorithm, baseline in param_grid:
        # Skip baseline = True for ActorCritic
        if algorithm == 'ActorCritic' and baseline:
            continue
        
        print(f"Training with learning_rate={lr}, gamma={gamma}, algorithm={algorithm}, baseline={baseline}")
        mean_reward, std_reward, agent = train(lr, gamma, algorithm, baseline, args.n_episodes)
        results.append((lr, gamma, algorithm, baseline, mean_reward, std_reward, agent))
        
    # Separate results for each algorithm
    reinforce_results = [result for result in results if result[2] == 'Reinforce' and not result[3]]
    reinforce_baseline_results = [result for result in results if result[2] == 'Reinforce' and result[3]]
    actor_critic_results = [result for result in results if result[2] == 'ActorCritic']
    
    # Determine best parameters for each algorithm
    best_reinforce_params = max(reinforce_results, key=lambda x: x[4])
    best_reinforce_baseline_params = max(reinforce_baseline_results, key=lambda x: x[4])
    best_actor_critic_params = max(actor_critic_results, key=lambda x: x[4])
    
    # Print best parameters
    print(f"Best parameters for Reinforce: learning_rate={best_reinforce_params[0]}, gamma={best_reinforce_params[1]}, baseline={best_reinforce_params[3]} with mean reward={best_reinforce_params[4]} +/- {best_reinforce_params[5]}")
    print(f"Best parameters for Reinforce with baseline: learning_rate={best_reinforce_baseline_params[0]}, gamma={best_reinforce_baseline_params[1]}, baseline={best_reinforce_baseline_params[3]} with mean reward={best_reinforce_baseline_params[4]} +/- {best_reinforce_baseline_params[5]}")
    print(f"Best parameters for ActorCritic: learning_rate={best_actor_critic_params[0]}, gamma={best_actor_critic_params[1]} with mean reward={best_actor_critic_params[4]} +/- {best_actor_critic_params[5]}")
    
    # Save the best models
    torch.save(best_reinforce_params[6].policy.state_dict(), "best_reinforce_model.mdl")
    torch.save(best_reinforce_baseline_params[6].policy.state_dict(), "best_reinforce_baseline_model.mdl")
    torch.save(best_actor_critic_params[6].policy.state_dict(), "best_actor_critic_model.mdl")

    # Plot the results for each algorithm
    def plot_results(results, label):
        mean_rewards = [result[4] for result in results]
        std_rewards = [result[5] for result in results]
        
        plt.errorbar(range(len(results)), mean_rewards, yerr=std_rewards, fmt='-o', capsize=5, label = label)
        
    plot_results(reinforce_results, 'Reinforce')
    plot_results(reinforce_baseline_results, 'Reinforce with Baseline')
    plot_results(actor_critic_results, 'ActorCritic')
           
    plt.xlabel('Hyperparameter combination index')
    plt.ylabel('Mean Reward')
    plt.title('Mean Rewards for different hyperparameters')
    
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
	main()