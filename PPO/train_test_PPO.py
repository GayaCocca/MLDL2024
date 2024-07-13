import gym
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def evaluate_model(model, env_id, n_eval_episodes=50):
    env = gym.make(env_id)
    env = Monitor(env)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, render=False)
    return mean_reward

def main():
    n_policies = 3  # Number of different policies to train for variance calculation
    n_eval_episodes = 50
    eval_interval = 1000 # Evaluate every 1000 episodes
    total_timesteps = 100000

    # Collect mean rewards for each training step
    source_rewards = {i: [] for i in range(eval_interval, total_timesteps + 1, eval_interval)}
    target_rewards = {i: [] for i in range(eval_interval, total_timesteps + 1, eval_interval)}

    for _ in range(n_policies):
        # Train on source environment
        train_env = gym.make('CustomHopper-source-v0')
        model = PPO('MlpPolicy', train_env, learning_rate=0.001, gamma=0.99, verbose=0)

        for step in range(eval_interval, total_timesteps + 1, eval_interval):
            model.learn(total_timesteps=eval_interval, reset_num_timesteps=False)
            mean_reward = evaluate_model(model, 'CustomHopper-target-v0', n_eval_episodes)
            source_rewards[step].append(mean_reward)

        # Train on target environment
        train2_env = gym.make('CustomHopper-target-v0')
        model2 = PPO('MlpPolicy', train2_env, learning_rate=0.001, gamma=0.99, verbose=0)

        for step in range(eval_interval, total_timesteps + 1, eval_interval):
            model2.learn(total_timesteps=eval_interval, reset_num_timesteps=False)
            mean_reward = evaluate_model(model2, 'CustomHopper-target-v0', n_eval_episodes)
            target_rewards[step].append(mean_reward)

    # Prepare data for plotting
    plot_data = []
    for step in source_rewards:
        for reward in source_rewards[step]:
            plot_data.append(['Source-Target', step, reward])
        for reward in target_rewards[step]:
            plot_data.append(['Target-Target', step, reward])
            

    np.save('PPOsource_results.npy', source_rewards)
    np.save('PPOtarget_results.npy', target_rewards)
    df = pd.DataFrame(plot_data, columns=['Environment', 'Timesteps', 'Mean Reward'])

    # Plot the results
    plt.figure(figsize=(12, 8))
    sns.lineplot(x='Timesteps', y='Mean Reward', hue='Environment', data=df, errorbar='sd')
    plt.title('Performance Comparison of PPO')
    plt.ylabel('Mean Reward')
    plt.xlabel('Training Timesteps')
    plt.show()

if __name__ == '__main__':
    main()
