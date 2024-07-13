import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def main():

    UDR_rewards = np.load('UDR_results.npy', allow_pickle=True).item()
    SIMOPT_rewards = np.load('SimOpt_results.npy', allow_pickle=True).item()
    PPOsource_rewards = np.load('PPOsource_results.npy', allow_pickle=True).item()
    PPOtarget_rewards = np.load('PPOtarget_results.npy', allow_pickle=True).item()

    plot_data = []
    for step in SIMOPT_rewards:
        for reward in UDR_rewards[step]:
            plot_data.append(['UDR', step, reward])
        for reward in SIMOPT_rewards[step]:
            plot_data.append(['SimOpt', step, reward])
        for reward in PPOsource_rewards[step]:
            plot_data.append(['PPOsource', step, reward])
        for reward in PPOtarget_rewards[step]:
            plot_data.append(['PPOtarget', step, reward])

    df = pd.DataFrame(plot_data, columns=['Environment', 'Timesteps', 'Mean Reward'])
    
    # Calculate mean rewards for UDR and SimOpt
    df_UDR = df[df['Environment'] == 'UDR'].groupby('Timesteps')['Mean Reward'].mean().reset_index()
    df_SimOpt = df[df['Environment'] == 'SimOpt'].groupby('Timesteps')['Mean Reward'].mean().reset_index()

    # Calculate bounds for PPOsource and PPOtarget
    df_PPOsource = df[df['Environment'] == 'PPOsource'].groupby('Timesteps')['Mean Reward'].mean().reset_index()
    df_PPOtarget = df[df['Environment'] == 'PPOtarget'].groupby('Timesteps')['Mean Reward'].mean().reset_index()

    # Ensure the timesteps match up for fill_between
    timesteps = df_UDR['Timesteps']
    PPOsource_bounds = df_PPOsource['Mean Reward']
    PPOtarget_bounds = df_PPOtarget['Mean Reward']
    
    # Plotting
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(12, 8))
    
    # Plot UDR and SimOpt
    plt.plot(df_UDR['Timesteps'], df_UDR['Mean Reward'], label='UDR', color='blue')
    plt.plot(df_SimOpt['Timesteps'], df_SimOpt['Mean Reward'], label='SimOpt', color='red')

    # Fill between PPOsource and PPOtarget
    plt.fill_between(timesteps, PPOsource_bounds, PPOtarget_bounds, color='gray', alpha=0.3, label='Bounds (PPOsource to PPOtarget)')

    plt.title('Performance Comparison between UDR and SimOpt')
    plt.ylabel('Mean Reward')
    plt.xlabel('Training Timesteps')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
