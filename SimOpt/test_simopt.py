import gym
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import itertools


#def parse_args():
 #   parser = argparse.ArgumentParser()
  #  parser.add_argument('--model', default='Simopt_ppo_policy', type=str, help='Model path')
   # parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    #parser.add_argument('--render', default=True, action='store_true', help='Render the simulator')
    #parser.add_argument('--episodes', default=50, type=int, help='Number of test episodes')
    #return parser.parse_args()

def main():
    #args = parse_args()

    model = PPO.load("Simopt_ppo_policy_final")
    real_env = gym.make('CustomHopper-target-v0')

    mean_reward, std_reward = evaluate_policy(model, real_env, n_eval_episodes=50, render = True)

    print(mean_reward, std_reward)

if __name__ == '__main__':
    main()