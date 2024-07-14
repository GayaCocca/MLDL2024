import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
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
    parser.add_argument('--print-every', default=200, type=int, help='Print info every <> episodes')
    parser.add_argument('--test-every', default=1000, type=int, help='Test every <> episodes')
    parser.add_argument('--test-episodes', default=10, type=int, help='Number of test episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--algorithm', default='Reinforce', choices=['Reinforce', 'ReinforceBaseline', 'ActorCritic'], help='Algorithm to use: Reinforce or ActorCritic')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2], help='List of seeds for training')
    return parser.parse_args()
args = parse_args()

import pandas as pd

#unofficial function
def plot_test_results4(test_rewards):
    # Set up the plotting
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    for algorithm, seed_rewards in test_rewards.items():
        episodes = np.arange(args.test_every, args.n_episodes + 1, args.test_every)
        rewards = np.array([seed_rewards[seed] for seed in args.seeds])

        # Calculate the mean and standard deviation across seeds
        mean_rewards = rewards.mean(axis=0)
        std_rewards = rewards.std(axis=0)

        # Plot the mean reward with standard deviation as the shaded area
        plt.plot(episodes, mean_rewards, label=f'{algorithm} Mean Reward')
        plt.fill_between(episodes, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2)

    plt.xlabel('Episodes')
    plt.ylabel('Mean Reward')
    plt.title('Mean Reward over Episodes for Different Algorithms')
    plt.legend()
    plt.show()

def test_policy(agent, env, args):
    rewards = []
    for _ in range(args.test_episodes):
        done = False
        test_reward = 0
        state = env.reset()
        while not done:
            action, _ = agent.get_action(state, evaluation=True)
            state, reward, done, _ = env.step(action.detach().cpu().numpy())
            test_reward += reward
        rewards.append(test_reward)
    return np.mean(rewards)

def plot_test_results(test_rewards, test_every):
    data = []
    for alg in test_rewards:
        for seed in args.seeds:
            for episode, reward in enumerate(test_rewards[alg][seed]):
                data.append({
                    'Algorithm': alg,
                    'Seed': seed,
                    'Episode': episode * test_every + test_every,  
                    'Reward': reward,
                })
       
    df = pd.DataFrame(data)
    #print(df)

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="darkgrid")
    sns.lineplot(data=df, x="Episode", y="Reward", hue="Algorithm", errorbar="sd")  # errorbar="sd" for std dev

    plt.xlabel('Training Episode')
    plt.ylabel('Average Test Reward')
    plt.title('Test Reward Comparison over Training')
    plt.legend()
    plt.show()



def main():
    env = gym.make('CustomHopper-source-v0') 
    # env = gym.make('CustomHopper-target-v0')  

    print('Action space:', env.action_space)
    print('State space:', env.observation_space)
    print('Dynamics parameters:', env.get_parameters())

    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    algorithms = ['Reinforce', 'ReinforceBaseline', 'ActorCritic']

    test_rewards = {alg: {seed: [] for seed in args.seeds} for alg in algorithms}

    for algorithm in algorithms:
        for seed in args.seeds:
            torch.manual_seed(seed) #all operations in Pytorch with the same seed have the same result
            np.random.seed(seed) #all operations in Numpy with the same seed have the same result
            env.seed(seed)

            policy = Policy(observation_space_dim, action_space_dim)
            agent = Agent(policy, device=args.device)
            episode_rewards = []  # Track rewards for the current algorithm

            for episode in range(args.n_episodes):
                done = False
                train_reward = 0

                state = env.reset()

                while not done:
                    action, action_probabilities = agent.get_action(state)
                    previous_state = state
                    state, reward, done, info = env.step(action.detach().cpu().numpy())
                    agent.store_outcome(previous_state, state, action_probabilities, reward, done)
                    train_reward += reward

                episode_rewards.append(train_reward)

                if (episode + 1) % args.print_every == 0:
                    print(f'Training episode: {episode} (Algorithm: {algorithm}, Seed: {seed})')
                    print(f'Episode return: {train_reward}')

                agent.update_policy(algorithm=algorithm)  # Update based on the specific algorithm

                if (episode + 1) % args.test_every == 0:
                    avg_reward = test_policy(agent, env, args)
                    test_rewards[algorithm][seed].append(avg_reward)

            # Save the final model for the current algorithm and seed
            torch.save(agent.policy.state_dict(), f"model_{algorithm}_seed{seed}.pth")
            print(np.mean(episode_rewards))
            
    #plot_test_results4(test_rewards)
    plot_test_results(test_rewards, args.test_every)

if __name__ == '__main__':
    main()
