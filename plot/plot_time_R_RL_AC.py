import argparse
import time
import numpy as np
import torch
import gym
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.distributions import Normal

from env.custom_hopper import *  # Assuming you have your custom Hopper env
from agent import Agent, Policy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=100000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=1000, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--algorithm', default='Reinforce', choices=['Reinforce', 'ReinforceBaseline', 'ActorCritic'], help='Algorithm to use: Reinforce or ActorCritic')
    return parser.parse_args()

args = parse_args()

def plot_mean_episode_time(mean_episode_times, intervals):
    data = []
    for alg in mean_episode_times:
        for interval, mean_time in zip(intervals, mean_episode_times[alg]):
            data.append({'Algorithm': alg, 'Episode': interval, 'Mean Episode Time (seconds)': mean_time})
    
    df = pd.DataFrame(data)
    print("DataFrame contents:")
    print(df.head())  # Print DataFrame to debug
    print(df.columns)  # Print column names
    print(df.dtypes)  # Print data types

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    sns.lineplot(data=df, x="Episode", y="Mean Episode Time (seconds)", hue="Algorithm")

    plt.xlabel('Training Episode')
    plt.ylabel('Mean Episode Time (seconds)')
    plt.title('Mean Episode Time over Training for Different Algorithms')
    plt.legend()
    plt.show()

def main():
    env = gym.make('CustomHopper-source-v0')
    
    print('Action space:', env.action_space)
    print('State space:', env.observation_space)
    print('Dynamics parameters:', env.get_parameters())

    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    algorithms = ['Reinforce', 'ReinforceBaseline', 'ActorCritic']
    interval = 20

    mean_episode_times = {alg: [] for alg in algorithms}
    intervals = list(range(interval, args.n_episodes + 1, interval))
    
    for algorithm in algorithms:
        torch.manual_seed(0)
        np.random.seed(0)
        env.seed(0)

        policy = Policy(observation_space_dim, action_space_dim)
        agent = Agent(policy, device=args.device)
        episode_times = []  # Track time for each episode

        for episode in range(args.n_episodes):
            start_episode_time = time.time()  # Start timer for the episode

            done = False
            train_reward = 0
            state = env.reset()

            while not done:
                action, action_probabilities = agent.get_action(state)
                previous_state = state
                state, reward, done, info = env.step(action.detach().cpu().numpy())
                agent.store_outcome(previous_state, state, action_probabilities, reward, done)
                train_reward += reward

            end_episode_time = time.time()  # End timer for the episode
            episode_time = end_episode_time - start_episode_time
            episode_times.append(episode_time)

            if (episode + 1) % args.print_every == 0:
                print(f'Training episode: {episode} (Algorithm: {algorithm})')
                print(f'Episode return: {train_reward}')

            agent.update_policy(algorithm=algorithm)  # Update based on the specific algorithm

            if (episode + 1) % interval == 0:
                mean_time_so_far = np.mean(episode_times)
                mean_episode_times[algorithm].append(mean_time_so_far)
                episode_times = []  # Reset episode times for the next interval

        # Save the final model for the current algorithm
        torch.save(agent.policy.state_dict(), f"model_{algorithm}.pth")

    plot_mean_episode_time(mean_episode_times, intervals)

if __name__ == '__main__':
    main()
