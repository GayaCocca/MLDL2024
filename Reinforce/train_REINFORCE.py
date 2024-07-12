"""Train an RL agent on the OpenAI Gym Hopper environment using
    REINFORCE algorithm
"""
import argparse

import matplotlib.pyplot as plt
import time
import torch
import gym

from env.custom_hopper import *
from agent import Agent, Policy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=100000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=200, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--algorithm', default='Reinforce', choices=['Reinforce', 'ActorCritic'], help='Algorithm to use: Reinforce or ActorCritic')

    return parser.parse_args()

args = parse_args()


def main():

	env = gym.make('CustomHopper-source-v0')
	# env = gym.make('CustomHopper-target-v0')

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())


	"""
		Training
	"""
	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	policy = Policy(observation_space_dim, action_space_dim)
	agent = Agent(policy, device=args.device)
	
    #
    # TASK 2 and 3: interleave data collection to policy updates
    #

	episode_rewards = []
	episode_times = []

	for episode in range(args.n_episodes):
		
		start_time = time.time()
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

		if (episode+1)%args.print_every == 0:
			print('Training episode:', episode)
			print('Episode return:', train_reward)

		agent.update_policy(algorithm=args.algorithm)
		end_time = time.time()
		episode_time = end_time - start_time

		episode_times.append(episode_time)
		
	torch.save(agent.policy.state_dict(), "model.Reinforce")
	print('mean reward over train:', np.mean(episode_rewards))

	plt.plot(episode_rewards)
	plt.xlabel('Episode')
	plt.ylabel('Train Reward')
	plt.title('Training Reward Over Episodes')
	plt.show()
	plt.plot(episode_times)
	plt.xlabel('Episode')
	plt.ylabel('Episode time')
	plt.title('Episode time Over Episodes')
	plt.show()

	

if __name__ == '__main__':
	main()