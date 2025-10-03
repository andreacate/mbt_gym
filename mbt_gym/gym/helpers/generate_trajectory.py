import gym
import numpy as np
import torch

from mbt_gym.agents.Agent import Agent
from mbt_gym.agents.SbAgent import SbAgent
from mbt_gym.gym.index_names import CASH_INDEX, INVENTORY_INDEX, ASSET_PRICE_INDEX, TIME_INDEX, FADS_INDEX, FILTERED_FADS_INDEX


# def generate_trajectory(env: gym.Env, agent: Agent, seed: int = None, include_log_probs: bool = False):
#     if seed is not None:
#         env.seed(seed)
#     obs_space_dim = env.observation_space.shape[0]
#     action_space_dim = env.action_space.shape[0]
#     observations = np.zeros((env.num_trajectories, obs_space_dim, env.n_steps + 1))
#     actions = np.zeros((env.num_trajectories, action_space_dim, env.n_steps))
#     rewards = np.zeros((env.num_trajectories, 1, env.n_steps))
#     if include_log_probs:
#         log_probs = torch.zeros((env.num_trajectories, env.action_space.shape[0], env.n_steps))
#     obs = env.reset()
#     observations[:, :, 0] = obs
#     count = 0
#     while True:
#         if include_log_probs:
#             action, log_prob = agent.get_action(obs, include_log_probs=True)
#         else:
#             action = agent.get_action(obs)
#         obs, reward, done, _ = env.step(action)
#         actions[:, :, count] = action
#         observations[:, :, count + 1] = obs
#         rewards[:, :, count] = reward.reshape(-1, 1)
#         if include_log_probs:
#             log_probs[:, :, count] = log_prob
#         if (env.num_trajectories > 1 and done[0]) or (env.num_trajectories == 1 and done):
#             break
#         count += 1
#     if include_log_probs:
#         return observations, actions, rewards, log_probs
#     else:
#         return observations, actions, rewards
    

def generate_trajectory(env: gym.Env, agent: Agent, seed: int = None, include_log_probs: bool = False):
    """Generate trajectory with dynamic observation shape detection"""
    if seed is not None:
        env.seed(seed)
        #agent.seed(seed)
    
    # Get initial observation to determine actual shape
    initial_obs = env.reset()
    if isinstance(initial_obs, tuple):
        initial_obs = initial_obs[0]  # Handle gym environments that return (obs, info)
    
    # Determine actual observation dimensions
    actual_obs_shape = initial_obs.shape
    num_trajectories = actual_obs_shape[0]
    obs_dim = actual_obs_shape[1]
    
    # Pre-allocate arrays with correct dimensions
    observations = np.zeros((num_trajectories, obs_dim, env.n_steps + 1))
    actions = np.zeros((num_trajectories, env.action_space.shape[0], env.n_steps))
    rewards = np.zeros((num_trajectories, 1, env.n_steps))
    # debugging
    # print("Obs shape",observations.shape)
    # print("Act shape",actions.shape)
    # print("Rewards shape", rewards.shape)
    
    if include_log_probs:
        log_probs = torch.zeros((num_trajectories, env.action_space.shape[0], env.n_steps))
    
    # Store initial observation
    observations[:, :, 0] = initial_obs
    count = 0
    
    while True:
        if include_log_probs:
            action, log_prob = agent.get_action(observations[:, :, count], include_log_probs=True)
            log_probs[:, :, count] = log_prob
        else:
            action = agent.get_action(observations[:, :, count])
        # if include_log_probs:
        #     action, log_prob = agent.get_action_and_log_prob(observations[:, :, count])
        #     log_probs[:, :, count] = log_prob
        # #print("action dim",agent.get_action(observations[:, :, count]))
        # action = agent.get_action(observations[:, :, count])
        
        actions[:, :, count] = action
        obs, reward, done, info = env.step(action)
        
        if isinstance(obs, tuple):
            obs = obs[0]  # Handle gym environments that return (obs, info)
            
        observations[:, :, count + 1] = obs
        rewards[:, :, count] = reward.reshape((-1, 1))
        count += 1
        
        if done[0] or count >= env.n_steps:
            break
    
    # Trim arrays to actual length
    observations = observations[:, :, :count + 1]
    actions = actions[:, :, :count]
    rewards = rewards[:, :, :count]

    if include_log_probs:
        log_probs = log_probs[:, :, :count]
        return observations, actions, rewards, log_probs
    else:
        return observations, actions, rewards
    

def generate_trajectory_rl(env: gym.Env, rl_agent: SbAgent, seed: int = None, include_log_probs: bool = False):
    """
    Generate trajectory for RL agents that need specific state transformations.
    Handles the case where the RL agent was trained on a reduced state space.
    """
    if seed is not None:
        env.seed(seed)
    
    # Get initial observation
    initial_obs = env.reset()
    if isinstance(initial_obs, tuple):
        initial_obs = initial_obs[0]  # Handle gym environments that return (obs, info)
    
    # Determine dimensions
    num_trajectories = env.num_trajectories
    obs_dim = initial_obs.shape[1]
    
    # Pre-allocate arrays
    observations = np.zeros((num_trajectories, obs_dim, env.n_steps + 1))
    actions = np.zeros((num_trajectories, env.action_space.shape[0], env.n_steps))
    rewards = np.zeros((num_trajectories, 1, env.n_steps))
    
    if include_log_probs:
        log_probs = torch.zeros((num_trajectories, env.action_space.shape[0], env.n_steps))
    
    # Store initial observation
    observations[:, :, 0] = initial_obs
    count = 0
    
    while True:
        # Get current observations for this step
        current_obs = observations[:, :, count]
        # print("Has reduced_training attribute:", hasattr(rl_agent, 'reduced_training'))
        # if hasattr(rl_agent, 'reduced_training'):
        #     print("reduced_training value:", rl_agent.reduced_training)
        # Transform observation for RL agent based on reduced_training_indices if applicable
        if hasattr(rl_agent, 'reduced_training') and rl_agent.reduced_training:
            # Use only the indices the agent was trained on
            # print("Reduce",rl_agent.reduced_training_indices)
            rl_obs = current_obs[:, rl_agent.reduced_training_indices]
        else:
            # Use full observation
            rl_obs = current_obs
            
        # Get action from RL agent using transformed observation
        if include_log_probs:
            action, log_prob = rl_agent.get_action(rl_obs, include_log_probs=True)
            log_probs[:, :, count] = log_prob
        else:
            action = rl_agent.get_action(rl_obs)
        
        # Step environment with action
        actions[:, :, count] = action
        obs, reward, done, info = env.step(action)
        
        if isinstance(obs, tuple):
            obs = obs[0]  # Handle gym environments that return (obs, info)
            
        observations[:, :, count + 1] = obs
        rewards[:, :, count] = reward.reshape((-1, 1))
        count += 1
        
        if done[0] or count >= env.n_steps:
            break
    
    # Trim arrays to actual length
    observations = observations[:, :, :count + 1]
    actions = actions[:, :, :count]
    rewards = rewards[:, :, :count]
    
    if include_log_probs:
        log_probs = log_probs[:, :, :count]
        return observations, actions, rewards, log_probs
    else:
        return observations, actions, rewards


# def generate_trajectory_rl_old(env: gym.Env, rl_agent: SbAgent, seed: int = None, include_log_probs: bool = False):
#     """
#     Generate trajectory for RL agents that need specific state transformations.
#     Handles the case where the RL agent was trained on a reduced state space.
#     """
#     if seed is not None:
#         env.seed(seed)
    
#     # Get initial observation
#     initial_obs = env.reset()
#     if isinstance(initial_obs, tuple):
#         initial_obs = initial_obs[0]  # Handle gym environments that return (obs, info)
    
#     # Determine dimensions
#     num_trajectories = env.num_trajectories
#     obs_dim = initial_obs.shape[1]
#     print("obs_dim",obs_dim)
    
#     # Pre-allocate arrays
#     observations = np.zeros((num_trajectories, obs_dim, env.n_steps + 1))
#     actions = np.zeros((num_trajectories, env.action_space.shape[0], env.n_steps))
#     rewards = np.zeros((num_trajectories, 1, env.n_steps))
    
#     if include_log_probs:
#         log_probs = torch.zeros((num_trajectories, env.action_space.shape[0], env.n_steps))
    
#     # Store initial observation
#     observations[:, :, 0] = initial_obs
#     print("observation",observations.shape)
#     count = 0
    
#     while True:
#         # Transform observation for RL agent - now using 4 dimensions
#         # [inventory, time, price, fads]
#         rl_obs = np.zeros((num_trajectories, 4))
#         for i in range(num_trajectories):
#             # Extract inventory
#             if INVENTORY_INDEX < obs_dim:
#                 rl_obs[i, 0] = observations[i, INVENTORY_INDEX, count]
            
#             # Add normalized time
#             if TIME_INDEX < obs_dim:
#                 rl_obs[i, 1] = observations[i, TIME_INDEX, count]
            
#             # Add price
#             if ASSET_PRICE_INDEX < obs_dim:
#                 rl_obs[i, 2] = observations[i, ASSET_PRICE_INDEX, count]
            
#             # Add fads component
#             # Assuming FADS_INDEX is defined - if not, you'll need to define it
#             if 'FADS_INDEX' in globals() and FADS_INDEX < obs_dim:
#                 rl_obs[i, 3] = observations[i, FADS_INDEX, count]
#             else:
#                 # Default to 0 if not available
#                 rl_obs[i, 3] = 0.0
        
#         # Get action from RL agent using transformed observation
#         if include_log_probs:
#             action, log_prob = rl_agent.get_action_and_log_prob(rl_obs)
#             log_probs[:, :, count] = log_prob
#         else:
#             action = rl_agent.get_action(rl_obs)
        
#         # Step environment with action
#         actions[:, :, count] = action
#         obs, reward, done, info = env.step(action)
        
#         if isinstance(obs, tuple):
#             obs = obs[0]  # Handle gym environments that return (obs, info)
            
#         observations[:, :, count + 1] = obs
#         rewards[:, :, count] = reward.reshape((-1, 1))
#         count += 1
        
#         if done[0] or count >= env.n_steps:
#             break
    
#     # Trim arrays to actual length
#     observations = observations[:, :, :count + 1]
#     actions = actions[:, :, :count]
#     rewards = rewards[:, :, :count]
    
#     if include_log_probs:
#         log_probs = log_probs[:, :, :count]
#         return observations, actions, rewards, log_probs
#     else:
#         return observations, actions, rewards