# Enhancing Sim-to-Real Transfer: Effective Reinforcement Learning for Robotic Controls
##### by Gaya Cocca, Arianna Coppola, Virginia Zura-Puntaroni.

- [Enhancing Sim to Real Transfer: Effective Reinforcement Learning for Robotic Controls](#enhancing-sim-to-real-transfer-effective-reinforcement-learning-for-robotic-controls)
				- [by Gaya Cocca, Arianna Coppola, Virginia Zura-Puntaroni.](#by-gaya-cocca-arianna-coppola-virginia-zura-puntaroni)
	- [Report Abstract](#report-abstract)
- [Requirements](#requirements)
- [Environment](#environment)
- [Algorithms](#algorithms)
	- [Reinforce](#reinforce)
		- [How to run the code](#how-to-run-the-code)
  - [REINFORCE_Baseline](#reinforce-baseline)
    - [How to run the code](#how-to-run-the-code-1)
  - [ACTOR_CRITIC](#actor-critic)
    - [How to run the code](#how-to-run-the-code-2)
  - [PPO](#ppo)
	  - [How to run the code](#how-to-run-the-code-3)
- [Tuning AC_R_RB](#tuning-ac-r-rb)
    - [How to run the code](#how-to-run-the-code-4)
- [Uniform Domain Randomization](#uniform-domain-randomization)
	- [How to run the code](#how-to-run-the-code-5)
- [SimOpt](#simopt)
	- [How to run the code](#how-to-run-the-code-6)
- [plot](#plot)
  - [How to run the code](#how-to-run-the-code-7)

The repository contains the code for project 4: *Reinforcement Learning* in the context of robotic systems.


## Report Abstract
This paper explores Reinforcement Learning in robotic
systems, with a focus on the sim-to-real transfer challenge.
Starting from the implementation of basic algorithms such as
REINFORCE, REINFORCE with baseline and Actor-Critic, the
robustness of the policies is improved by the integration of
Proximal Policy Optimization (PPO) combined with Uniform
Domain Randomization.
The project’s key extension involves implementing Adaptive 
Domain Randomization, specifically SimOpt, to further enhance the
robustness and transferability of these policies from simulation
to real-world environments. By analyzing and applying these
methodologies, the paper aims to find concrete solutions to the
Reality Gap problem.



# Requirements
- [Mujoco-py](https://github.com/openai/mujoco-py)
- [Gym](https://github.com/openai/gym)
- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)



# Environment
The environment used to train the agent is the [Hopper environment](https://www.gymlibrary.dev/environments/mujoco/hopper/) from the Gym API. 
Hopper is one-legged robot which underlying physics engine is modeled with MuJoCo.
The goal of the Hopper is to learn how to jump in the forward direction without falling,
while achieving the highest possible horizontal speed. 
This is accomplished by applying torques on the three hinges connecting the four body parts.
By running the `test_random_policy.py` script, it is possible to find out some of the key feature of the environment.



# Algorithms
The repository contains the implementation of different Reinforcement Learning algorithms, an implementation of Uniform Domain Randomization combined with PPO and one of SimOpt.



## Reinforce
Implementation of the basic REINFORCE algorithm.

### How to run the code
- `agent.py` contains the implementation of the agent and the policy classes to use to train the policy with the REINFROCE algorithm.
- Running `train_REINFORCE.py` starts a training by episodes on the source environment, with the possibility to:
  - select the number of training episodes;
  - select how often to print the training information (in terms of episodes);
  - select the device.
    
  The resulting model is saved in `model.Reinforce`.
- Running `test_reinforce.py` starts a test by episodes on the source environment, with the possibility to select the number of test episodes.




## REINFORCE_Baseline
Implementation of REINFORCE algorithm with the introduction of a baseline.

### How to run the code
- `agent.py` contains the implementation of the agent and the policy classes to use to train the policy with the REINFORCE with baseline algorithm. It is possible to change the value of the baseline directly in this code.
- Running `train_REINFORCE_BAS.py` starts a training by episodes on the source environment, with the possibility to:
  - select the number of training episodes;
  - select how often to print the training information (in terms of episodes);
  - select the device.
    
  The resulting model is saved in `model.ReinforceBAS`.
- Running `test_REINFORCE.Baseline.py` starts a test by episodes on the source environment, with the possibility to select the number of test episodes.




## ACTOR_CRITIC
Implementation of the Actor Critic algorithm.

### How to run the code
- `agent.py` contains the implementation of the agent and the policy classes to train the policy with Actor Critic algorithm.
- Running `train_actor_critic.py` starts a training by episodes on the source environment, with the possibility to:
  - select the number of training episodes;
  - select how often to print the training information (in terms of episodes);
  - select the device.
- Running `test_actor_critic.py` starts a test by episodes on the source environment, with the possibility to select the number of test episodes.





## PPO
The implementation of the PPO algorithm is done with the help of a third-party library [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) (sb3).
### How to run the code
- Run `tuning_PPO.py` to pursue the tuning of learning rates and gammas. The best parameters are returned.
- In `train_test_PPO.py` two models are trained: one in the simulation environment, the other one on the target environment.
  Each of them is tested on target to understand how much the `source-target` pair underperforms the `target-target` pair.

  The two resulting models are stored in `model_PPO_source` and `model_PPO_target`.

PPO is used to implement the Uniform Domain Randomization.





## Tuning AC_R_RB
Tuning over different values of gamma and learning rate to get the best configuration of REINFORCE, REINFORCE with baseline and Actor-Critic algorithms.

### How to run the code
- `agent_TUNING.py` contains the implementation of the agent and policy classes used to train the policy with the different algorithms, properly modified for tuning.
- `train_tuningphase.py` performs a grid search to tune different values of gamma and learning rate, and then uses these configurations to start a training phase.

The resulting models are stored in `best_reinforce_model.mdl`, `best_reinforce_baseline_model.mdl` and `best_actor_critic_model.mdl`.







## UDR
To define Uniform Domain Randomization, new randomized environments are specified in `custom_hopper_grid.py` within the `env` folder: `CustomHopperRand-source-v0` and `CustomHopperRand-target-v0`.

### How to run the code
- Running `tuning_UDR.py` initiates a grid search to tune different mass ranges, which define the bounds of the uniform parameter distributions. The results of the tuning are saved in `best_mass_ranges.npy` and then used in `train_UDR.py`.
- Running `train_UDR.py` starts a training phase on the randomized source environment followed by a test phase on the target environment.

The resulting model is stored in the `PPO_mld_UDR` folder.






# SimOpt
To overcome some of the issues implied by UDR, another technique has been proposed. 
SimOpt was chosen from the Adaptive Domain Randomization methods. 
### How to run the code
- `train_simopt.py` run the PPO algorithm and train a new policy.
- `test_simopt.py` performs the test phase.

The resulting model is stored in the `Simopt_ppo_policy_final` folder.




# plot
Inside the plot folder are included the final comparison graph `final_plot_UDR_SIMOPT_BOUNDS.py` and the following results necessary for the first plot:
- `PPOsource_results.npy`.
- `PPOtarget_results.npy`.
- `SimOpt_results.npy`.
- `UDR_results.npy`.
In addiction, in the folder there are the performance comparison plot and the time comparison plot, respectively `final_plot_UDR_SIMOPT_BOUNDS.py` and `plot_time_R_RB_AC.py` between REINFORCE, REINFORCE + baseline and Actor Critic.
### How to run the code
`final_plot_UDR_SIMOPT_BOUNDS.py` shows the comparison plot between performances of the SimOpt and UDR models in relation with PPO boundaries.
`final_plot_UDR_SIMOPT_BOUNDS.py` shows the comparison between the performances of REINFORCE, REINFORCE + baseline and Actor Critic. Three different policies for each of them are trained and tested every 1000 episodes (for a total of 100000 episodes).
`plot_time_R_RB_AC.py` shows the comparison in time of REINFORCE, REINFORCE + baseline and Actor Critic. The average length of each episode is considered for each algorithm.











