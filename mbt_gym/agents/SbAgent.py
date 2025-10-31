import numpy as np

from mbt_gym.agents.Agent import Agent

from stable_baselines3.common.base_class import BaseAlgorithm
#from mbt_gym.gym.TradingEnvironment import TradingEnvironment


class SbAgent(Agent):
    def __init__(self, model: BaseAlgorithm, reduced_training_indices: list = None,  env = None, use_normalized_agent: bool = False, normalize_obs: bool = False):
        self.model = model
        self.env = env
        self.num_trajectories = self.env.num_trajectories
        self.num_actions = self.model.action_space.shape[0]
        self.use_normalized_agent = use_normalized_agent
        self.normalize_obs = normalize_obs
        if reduced_training_indices is not None:
            self.reduced_training = True
            self.reduced_training_indices = reduced_training_indices
        else:
            self.reduced_training = False

    # def get_action(self, state: np.ndarray) -> np.ndarray:
    #     # if self.reduced_training:
    #     #     state = state[:, self.reduced_training_indices]
    #     # return self.model.predict(state, deterministic=True)[0].reshape(self.num_trajectories, self.num_actions)
    #     #print("state dim", state.shape[0])
    #     action = self.model.predict(state, deterministic=True)[0].reshape(state.shape[0], self.num_actions)
    #     if self.env.normalise_action_space_:         
    #         #print("action before denormalisation", action)
    #         action = self.env.normalise_action(action, inverse=True)
    #     return action
    
    def get_action(self, state: np.ndarray) -> np.ndarray:
        # TODO: handle reduced training state space if needed (attention to the dimension reduced state or not?)
        # If the current env is NOT normalized but the model was trained on normalized obs,
        # we need to normalize the state before passing it to the model
        # if not self.env.normalise_observation_space_ and self.normalize_obs and hasattr(self.env, 'normalise_observation'):
        #     # Normalize observations for the model
        #     state = self.env.normalise_observation(state, inverse=False, force=True)
        #     print("check2")

        # Get action from model
        action = self.model.predict(state, deterministic=True)[0].reshape(state.shape[0], self.num_actions)
        
        # If the current env (for evaluation) is NOT normalized but the model outputs normalized actions,
        # we need to denormalize them
        if not self.env.normalise_action_space_ and self.use_normalized_agent and hasattr(self.env, 'normalise_action'):
            #print("check")
            action = self.env.normalise_action(action, inverse=True, force=True)
            
        return action

    def train(self, total_timesteps: int = 100000):
        self.model.learn(total_timesteps=total_timesteps)

    def get_state(self):
        return self.model.policy.state_dict() 

