from copy import deepcopy
from doctest import Example
from time import time

import gym
import numpy as np
import warnings
from scipy.linalg import expm
from scipy.integrate import quad

import matplotlib.pyplot as plt   

from mbt_gym.agents.Agent import Agent
from mbt_gym.gym.TradingEnvironment import TradingEnvironment
from mbt_gym.gym.index_names import FILTERED_FADS_INDEX, INVENTORY_INDEX, TIME_INDEX, ASSET_PRICE_INDEX, CASH_INDEX, BID_INDEX, ASK_INDEX, FADS_INDEX
from mbt_gym.rewards.RewardFunctions import CjMmCriterion, PnL
from mbt_gym.stochastic_processes.arrival_models import *
from mbt_gym.stochastic_processes.price_impact_models import PriceImpactModel, TemporaryAndPermanentPriceImpact
from mbt_gym.gym.ModelDynamics import LimitOrderModelDynamics, TradinghWithSpeedModelDynamics

from scipy.integrate import solve_ivp

class RandomAgent(Agent):
    def __init__(self, env: gym.Env, seed: int = None):
        self.action_space = deepcopy(env.action_space)
        self.action_space.seed(seed)
        self.num_trajectories = env.num_trajectories

    def get_action(self, state: np.ndarray) -> np.ndarray:
        return np.repeat(self.action_space.sample().reshape(1, -1), self.num_trajectories, axis=0)


class FixedActionAgent(Agent):
    def __init__(self, fixed_action: np.ndarray, env: gym.Env):
        self.fixed_action = fixed_action
        self.env = env

    def get_action(self, state: np.ndarray) -> np.ndarray:
        return np.repeat(self.fixed_action.reshape(1, -1), self.env.num_trajectories, axis=0)


class FixedSpreadAgent(Agent):
    def __init__(self, env: gym.Env, half_spread: float = 1.0, offset: float = 0.0):
        self.half_spread = half_spread
        self.offset = offset
        self.env = env

    def get_action(self, state: np.ndarray) -> np.ndarray:
        action = np.array([[self.half_spread - self.offset, self.half_spread + self.offset]])
        return np.repeat(action, self.env.num_trajectories, axis=0)


class HumanAgent(Agent):
    def get_action(self, state: np.ndarray):
        bid = float(input(f"Current state is {state}. How large do you want to set midprice-bid half spread? "))
        ask = float(input(f"Current state is {state}. How large do you want to set ask-midprice half spread? "))
        return np.array([bid, ask])


class AvellanedaStoikovAgent(Agent):
    def __init__(self, risk_aversion: float = 0.1, env: TradingEnvironment = None):
        self.risk_aversion = risk_aversion
        self.env = env or TradingEnvironment()
        assert isinstance(self.env, TradingEnvironment)
        self.terminal_time = self.env.terminal_time
        self.volatility = self.env.model_dynamics.midprice_model.volatility
        self.rate_of_arrival = self.env.model_dynamics.arrival_model.intensity
        self.fill_exponent = self.env.model_dynamics.fill_probability_model.fill_exponent

    def get_action(self, state: np.ndarray):
        inventory = state[:, INVENTORY_INDEX]
        time = state[:, TIME_INDEX]
        action = self._get_action(inventory, time)
        if action.min() < 0:
            warnings.warn("Avellaneda-Stoikov agent is quoting a negative spread")
        return action

    def _get_price_adjustment(self, inventory: int, time: float) -> float:
        return inventory * self.risk_aversion * self.volatility**2 * (self.terminal_time - time)

    def _get_spread(self, time: float) -> float:
        if self.risk_aversion == 0:
            return 2 / self.fill_exponent  # Limit as risk aversion -> 0
        volatility_aversion_component = self.risk_aversion * self.volatility**2 * (self.terminal_time - time)
        fill_exponent_component = 2 / self.risk_aversion * np.log(1 + self.risk_aversion / self.fill_exponent)
        return volatility_aversion_component + fill_exponent_component

    def _get_action(self, inventory: int, time: float):
        bid_half_spread = (self._get_price_adjustment(inventory, time) + self._get_spread(time) / 2).reshape(-1, 1)
        ask_half_spread = (-self._get_price_adjustment(inventory, time) + self._get_spread(time) / 2).reshape(-1, 1)
        return np.append(bid_half_spread, ask_half_spread, axis=1)
    
class ModifiedAvellanedaStoikovAgent(Agent):
    def __init__(self, risk_aversion: float = 0.1, env: TradingEnvironment = None):
        self.risk_aversion = risk_aversion
        self.env = env or TradingEnvironment()
        assert isinstance(self.env, TradingEnvironment)
        self.terminal_time = self.env.terminal_time
        self.volatility = self.env.model_dynamics.midprice_model.volatility
        arrival_model = self.env.model_dynamics.arrival_model
        self.rate_of_arrival = (
            getattr(arrival_model, "intensity", None)
            if getattr(arrival_model, "intensity", None) is not None
            else getattr(arrival_model, "current_state", None)
            )
        # print("type(self.rate_of_arrival):", type(self.rate_of_arrival), "self.rate_of_arrival:", self.rate_of_arrival.shape, self.rate_of_arrival)
        #self.rate_of_arrival = self.env.model_dynamics.arrival_model.intensity 
        self.fill_exponent = self.env.model_dynamics.fill_probability_model.fill_exponent

    def get_action(self, state: np.ndarray):
        inventory = state[:, INVENTORY_INDEX]
        time = state[:, TIME_INDEX]
        action = self._get_action(inventory, time)
        if action.min() < 0:
            warnings.warn("Avellaneda-Stoikov agent is quoting a negative spread")
        return action

    def _get_price_adjustment(self, inventory: int, time: float) -> float:
        return inventory * self.risk_aversion * self.volatility**2 * (self.terminal_time - time)  # 3.10 equation

    def _get_spread(self, time: float) -> float:
        if self.risk_aversion == 0:
            return 2 / self.fill_exponent  # Limit as risk aversion -> 0
        # the volatility component is already considered inside the _get_price_adjustment, so I just set to 0 since otherwise I will consider twice
        #volatility_aversion_component = self.risk_aversion * self.volatility**2 * (self.terminal_time - time)
        volatility_aversion_component = 0
        fill_exponent_component = 2 / self.risk_aversion * np.log(1 + self.risk_aversion / self.fill_exponent) #3.12 equation   
        return volatility_aversion_component + fill_exponent_component

    def _get_action(self, inventory: int, time: float):
        bid_half_spread = (self._get_price_adjustment(inventory, time) + self._get_spread(time) / 2).reshape(-1, 1)
        ask_half_spread = (-self._get_price_adjustment(inventory, time) + self._get_spread(time) / 2).reshape(-1, 1)
        return np.append(bid_half_spread, ask_half_spread, axis=1)


class CarteaJaimungalMmAgent(Agent): # market-making agent in a limit order book
    def __init__(
        self,
        env: TradingEnvironment = None,
    ):
        self.env = env or TradingEnvironment()
        assert isinstance(self.env.model_dynamics, LimitOrderModelDynamics), "Trader must be type LimitOrderTrader"
        assert isinstance(self.env.reward_function, (CjMmCriterion, PnL)), "Reward function for CjMmAgent is incorrect."
        # TODO attention here, what is the fill exponent if we have FadsInformedUniformedTradersArrivalModel?
        self.kappa = self.env.model_dynamics.fill_probability_model.fill_exponent
        self.num_trajectories = self.env.num_trajectories
        if isinstance(self.env.reward_function, PnL):
            self.inventory_neutral = True
            self.risk_neutral_action = 1 / self.kappa * np.ones((env.num_trajectories, env.action_space.shape[0]))
        else:
            self.inventory_neutral = False
            self.phi = env.reward_function.per_step_inventory_aversion
            self.alpha = env.reward_function.terminal_inventory_aversion
            assert self.env.reward_function.inventory_exponent == 2.0, "Inventory exponent must be = 2."
            self.terminal_time = self.env.terminal_time
            #self.lambdas = self.env.model_dynamics.arrival_model.intensity
            arrival_model = self.env.model_dynamics.arrival_model
            self.lambdas = (
            getattr(arrival_model, "intensity", None)
            if getattr(arrival_model, "intensity", None) is not None
            else getattr(arrival_model, "current_state", None)
            )
            self.lambdas = self.lambdas.reshape(-1, 1)
            #print("self.lambdas:", self.lambdas)
            self.max_inventory = env.max_inventory
            #self.a_matrix, self.z_vector = self._calculate_a_and_z()
            self.large_depth = 10_000
            # print("--- Original Agent ---")
            # print(f"Lambdas: {self.lambdas.flatten()}")
            #print(f"A-Matrix (top-left 3x3):\n{self.a_matrix[:3, :3]}")


    def get_action(self, state: np.ndarray):
        if self.inventory_neutral:
            return self.risk_neutral_action
        else:
            current_time = state[0, TIME_INDEX]
            inventories = state[:, INVENTORY_INDEX]
            # --- START OF FIX ---
            # Get the CURRENT arrival intensity from the environment at each step
            arrival_model = self.env.model_dynamics.arrival_model
            current_lambdas = (
                getattr(arrival_model, "intensity", None)
                if getattr(arrival_model, "intensity", None) is not None
                else getattr(arrival_model, "current_state", None)
            )
            current_lambdas = current_lambdas.reshape(-1, 1)

            # Recalculate the A-matrix and z-vector with the current lambdas
            a_matrix, z_vector = self._calculate_a_and_z(current_lambdas) # Pass lambdas as an argument
            # --- END OF FIX ---
            
            return self._calculate_deltas(
                inventories=inventories, 
                current_time=current_time,
                a_matrix=a_matrix, # Pass the newly calculated matrix
                z_vector=z_vector, # Pass the newly calculated vector
            )

            # assert (
            #     state[0, TIME_INDEX] == state[-1, TIME_INDEX]
            # ), "CarteaJaimungalMmAgent needs to be called on a tensor with a uniform time stamp."
            # current_time = state[0, TIME_INDEX]
            # inventories = state[:, INVENTORY_INDEX]
            # return self._calculate_deltas(inventories=inventories, current_time=current_time)

    def _calculate_deltas(self, current_time: float, inventories: np.ndarray, a_matrix: np.ndarray, z_vector: np.ndarray):
        deltas = np.zeros(shape=(self.num_trajectories, 2))
        h_t = self._calculate_ht(current_time, a_matrix, z_vector)
        # If the inventory goes above the max level, we quote a large depth to bring it back and quote on the opposite
        # side as if we had an inventory equal to sign(inventory) * self.max_inventory.
        indices = np.clip(self.max_inventory + inventories, 0, 2 * self.max_inventory)
        indices = indices.astype(int)
        indices_minus_one = np.clip(indices - 1, 0, 2 * self.max_inventory)
        indices_plus_one = np.clip(indices + 1, 0, 2 * self.max_inventory)
        h_0 = h_t[indices]
        h_plus_one = h_t[indices_plus_one]
        h_minus_one = h_t[indices_minus_one]
        max_inventory_bid = h_plus_one == h_0
        max_inventory_ask = h_minus_one == h_0
        deltas[:, BID_INDEX] = (1 / self.kappa - h_plus_one + h_0 + self.large_depth * max_inventory_bid).reshape(-1)
        deltas[:, ASK_INDEX] = (1 / self.kappa - h_minus_one + h_0 + self.large_depth * max_inventory_ask).reshape(-1)
        return deltas

    def _calculate_ht(self, current_time: float, a_matrix: np.ndarray, z_vector: np.ndarray) -> float:
        omega_function = self._calculate_omega(current_time, a_matrix, z_vector)
        return 1 / self.kappa * np.log(omega_function)

    def _calculate_omega(self, current_time: float, a_matrix: np.ndarray, z_vector: np.ndarray):
        """This is Equation (10.11) from [CJP15]."""
        return np.matmul(expm(a_matrix * (self.terminal_time - current_time)), z_vector)

    def _calculate_a_and_z(self, lambdas: np.ndarray):
        matrix_size = 2 * self.max_inventory + 1
        Amatrix = np.zeros(shape=(matrix_size, matrix_size))
        z_vector = np.zeros(shape=(matrix_size, 1))
        for i in range(matrix_size):
            inventory = self.max_inventory - i
            Amatrix[i, i] = -self.phi * self.kappa * inventory**2
            z_vector[i, 0] = np.exp(-self.alpha * self.kappa * inventory**2)
            if i + 1 < matrix_size:
                Amatrix[i, i + 1] = lambdas[BID_INDEX] * np.exp(-1)
            if i > 0:
                Amatrix[i, i - 1] = lambdas[ASK_INDEX] * np.exp(-1)
        return Amatrix, z_vector
    
    def calculate_true_value_function(self, state: np.ndarray):
        current_time = state[0, TIME_INDEX]
        inventories = state[:, INVENTORY_INDEX]
        value_fct = np.zeros(shape=(self.num_trajectories, 1))
        h_t = self._calculate_ht(current_time)
        indices = np.clip(self.max_inventory + inventories, 0, 2 * self.max_inventory)
        indices = indices.astype(int)
        h_0 = h_t[indices]
        value_fct = h_0 + state[:, CASH_INDEX] + state[:, INVENTORY_INDEX] * state[:, ASSET_PRICE_INDEX]
        return value_fct

class OriginalCarteaJaimungalMmAgent(Agent): 
    def __init__(
        self,
        env: TradingEnvironment = None,
        max_inventory: int = 100,
    ):
        self.env = env or TradingEnvironment()
        assert isinstance(self.env.model_dynamics, LimitOrderModelDynamics), "Trader must be type LimitOrderTrader"
        assert isinstance(self.env.reward_function, (CjMmCriterion, PnL)), "Reward function for CjMmAgent is incorrect."
        self.kappa = self.env.model_dynamics.fill_probability_model.fill_exponent
        self.num_trajectories = self.env.num_trajectories
        if isinstance(self.env.reward_function, PnL):
            self.inventory_neutral = True
            self.risk_neutral_action = 1 / self.kappa * np.ones((env.num_trajectories, env.action_space.shape[0]))
        else:
            self.inventory_neutral = False
            self.phi = env.reward_function.per_step_inventory_aversion
            self.alpha = env.reward_function.terminal_inventory_aversion
            assert self.env.reward_function.inventory_exponent == 2.0, "Inventory exponent must be = 2."
            self.terminal_time = self.env.terminal_time
            self.lambdas = np.array([[30.0], [30.0]])
            #self.lambdas = self.env.model_dynamics.arrival_model.intensity
            self.max_inventory = max_inventory
            self.a_matrix, self.z_vector = self._calculate_a_and_z()
            self.large_depth = 10_000

    def get_action(self, state: np.ndarray):
        if self.inventory_neutral:
            return self.risk_neutral_action
        else:
            assert (
                state[0, TIME_INDEX] == state[-1, TIME_INDEX]
            ), "CarteaJaimungalMmAgent needs to be called on a tensor with a uniform time stamp."
            current_time = state[0, TIME_INDEX]
            inventories = state[:, INVENTORY_INDEX]
            return self._calculate_deltas(inventories=inventories, current_time=current_time)

    def _calculate_deltas(self, current_time: float, inventories: np.ndarray):
        deltas = np.zeros(shape=(self.num_trajectories, 2))
        h_t = self._calculate_ht(current_time)
        # If the inventory goes above the max level, we quote a large depth to bring it back and quote on the opposite
        # side as if we had an inventory equal to sign(inventory) * self.max_inventory.
        indices = np.clip(self.max_inventory + inventories, 0, 2 * self.max_inventory)
        indices = indices.astype(int)
        indices_minus_one = np.clip(indices - 1, 0, 2 * self.max_inventory)
        indices_plus_one = np.clip(indices + 1, 0, 2 * self.max_inventory)
        h_0 = h_t[indices]
        h_plus_one = h_t[indices_plus_one]
        h_minus_one = h_t[indices_minus_one]
        max_inventory_bid = h_plus_one == h_0
        max_inventory_ask = h_minus_one == h_0
        deltas[:, BID_INDEX] = (1 / self.kappa - h_plus_one + h_0 + self.large_depth * max_inventory_bid).reshape(-1)
        deltas[:, ASK_INDEX] = (1 / self.kappa - h_minus_one + h_0 + self.large_depth * max_inventory_ask).reshape(-1)
        return deltas

    def _calculate_ht(self, current_time: float) -> float:
        omega_function = self._calculate_omega(current_time)
        # print("omega_function:", omega_function.flatten())
        return 1 / self.kappa * np.log(omega_function)

    def _calculate_omega(self, current_time: float):
        """This is Equation (10.11) from [CJP15]."""
        # print("self.a_matrix:", self.a_matrix)
        # print("self.z_vector:", self.z_vector.flatten())
        # print("self.terminal_time - current_time:", self.terminal_time - current_time)
        # print("expm(self.a_matrix * (self.terminal_time - current_time)):", expm(self.a_matrix * (self.terminal_time - current_time)))
        return np.matmul(expm(self.a_matrix * (self.terminal_time - current_time)), self.z_vector)

    def _calculate_a_and_z(self):
        matrix_size = 2 * self.max_inventory + 1
        Amatrix = np.zeros(shape=(matrix_size, matrix_size))
        z_vector = np.zeros(shape=(matrix_size, 1))
        # Add debug prints
        # print(f"alpha: {self.alpha}, kappa: {self.kappa}, phi: {self.phi}")
        for i in range(matrix_size):
            inventory = self.max_inventory - i
            Amatrix[i, i] = -self.phi * self.kappa * inventory**2
            z_term = -self.alpha * self.kappa * inventory**2
            z_vector[i, 0] = np.exp(z_term)
            # print(f"i: {i}, inventory: {inventory}, Amatrix[{i},{i}]: {Amatrix[i,i]}, z_vector[{i},0]: {z_vector[i,0]}")
            if i + 1 < matrix_size:
                Amatrix[i, i + 1] = self.lambdas[BID_INDEX] * np.exp(-1)
            if i > 0:
                Amatrix[i, i - 1] = self.lambdas[ASK_INDEX] * np.exp(-1)
        return Amatrix, z_vector
    
    def calculate_true_value_function(self, state: np.ndarray):
        current_time = state[0, TIME_INDEX]
        inventories = state[:, INVENTORY_INDEX]
        value_fct = np.zeros(shape=(self.num_trajectories, 1))
        h_t = self._calculate_ht(current_time)
        indices = np.clip(self.max_inventory + inventories, 0, 2 * self.max_inventory)
        indices = indices.astype(int)
        h_0 = h_t[indices]
        value_fct = h_0 + state[:, CASH_INDEX] + state[:, INVENTORY_INDEX] * state[:, ASSET_PRICE_INDEX]
        return value_fct


class ModifiedCarteaJaimungalMmAgent(Agent): # market-making agent in a limit order book
    def __init__(
        self,
        env: TradingEnvironment = None,
    ):
        self.env = env or TradingEnvironment()
        assert isinstance(self.env.model_dynamics, LimitOrderModelDynamics), "Trader must be type LimitOrderTrader"
        assert isinstance(self.env.reward_function, (CjMmCriterion, PnL)), "Reward function for CjMmAgent is incorrect."
        self.kappa = self.env.model_dynamics.fill_probability_model.fill_exponent
        self.num_trajectories = self.env.num_trajectories
        if isinstance(self.env.reward_function, PnL):
            self.inventory_neutral = True
            self.risk_neutral_action = 1 / self.kappa * np.ones((env.num_trajectories, env.action_space.shape[0]))
        else:
            self.inventory_neutral = False
            self.big_phi = env.reward_function.per_step_inventory_aversion
            self.alpha = env.reward_function.terminal_inventory_aversion
            assert self.env.reward_function.inventory_exponent == 2.0, "Inventory exponent must be = 2."
            self.terminal_time = self.env.terminal_time
            # arrival_model = self.env.model_dynamics.arrival_model
            self.gamma = self.env.model_dynamics.arrival_model.gamma
            self.eta = self.env.model_dynamics.midprice_model.eta
            self.phi = self.env.model_dynamics.arrival_model.phi
            self.psi = self.env.model_dynamics.arrival_model.psi
            self.sigma = self.env.model_dynamics.midprice_model.volatility
            self.fads_proportion =  self.env.model_dynamics.midprice_model.fads_proportion
            self.call_kappa = self.phi + self.psi / self.terminal_time * self._compute_integral()

            self.lambdas = np.array([[self.call_kappa], [self.call_kappa]])
            #print("self.lambdas:", self.lambdas)
            self.max_inventory = env.max_inventory
            self.a_matrix, self.z_vector = self._calculate_a_and_z()
            self.large_depth = 10_000
            # print("--- Modified Agent ---")
            # print(f"Integral value: {self._compute_integral()}")
            # print(f"self.phi: {self.phi}, self.psi: {self.psi}")
            # print(f"call_kappa: {self.call_kappa}")
            # print(f"Lambdas: {self.lambdas.flatten()}")
            # #self.a_matrix, self.z_vector = self._calculate_a_and_z()
            # print(f"A-Matrix (top-left 3x3):\n{self.a_matrix[:3, :3]}")

    def get_action(self, state: np.ndarray):
        if self.inventory_neutral:
            return self.risk_neutral_action
        else:
            assert (
                state[0, TIME_INDEX] == state[-1, TIME_INDEX]
            ), "CarteaJaimungalMmAgent needs to be called on a tensor with a uniform time stamp."
            current_time = state[0, TIME_INDEX]
            inventories = state[:, INVENTORY_INDEX]
            return self._calculate_deltas(inventories=inventories, current_time=current_time)

    def _calculate_deltas(self, current_time: float, inventories: np.ndarray):
        deltas = np.zeros(shape=(self.num_trajectories, 2))
        h_t = self._calculate_ht(current_time)
        # If the inventory goes above the max level, we quote a large depth to bring it back and quote on the opposite
        # side as if we had an inventory equal to sign(inventory) * self.max_inventory.
        indices = np.clip(self.max_inventory + inventories, 0, 2 * self.max_inventory)
        indices = indices.astype(int)
        indices_minus_one = np.clip(indices - 1, 0, 2 * self.max_inventory)
        indices_plus_one = np.clip(indices + 1, 0, 2 * self.max_inventory)
        h_0 = h_t[indices]
        h_plus_one = h_t[indices_plus_one]
        h_minus_one = h_t[indices_minus_one]
        max_inventory_bid = h_plus_one == h_0
        max_inventory_ask = h_minus_one == h_0
        deltas[:, BID_INDEX] = (1 / self.kappa - h_plus_one + h_0 + self.large_depth * max_inventory_bid).reshape(-1)
        deltas[:, ASK_INDEX] = (1 / self.kappa - h_minus_one + h_0 + self.large_depth * max_inventory_ask).reshape(-1)
        return deltas

    def _calculate_ht(self, current_time: float) -> float:
        omega_function = self._calculate_omega(current_time)
        if np.any(omega_function <= 0):
            print(f"[Warning] omega_function hit non-positive value at t={current_time}")   
        omega_function = np.maximum(omega_function, 1e-12)
        return 1 / self.kappa * np.log(omega_function)

    def _calculate_omega(self, current_time: float):
        """This is Equation (10.11) from [CJP15]."""
        return np.matmul(expm(self.a_matrix * (self.terminal_time - current_time)), self.z_vector)

    def _calculate_a_and_z(self):
        matrix_size = 2 * self.max_inventory + 1
        Amatrix = np.zeros(shape=(matrix_size, matrix_size))
        z_vector = np.zeros(shape=(matrix_size, 1))
        for i in range(matrix_size):
            inventory = self.max_inventory - i
            Amatrix[i, i] = -self.big_phi * self.kappa * inventory**2
            z_vector[i, 0] = np.exp(-self.alpha * self.kappa * inventory**2)
            if i + 1 < matrix_size:
                Amatrix[i, i + 1] = self.lambdas[BID_INDEX] * np.exp(-1)
            if i > 0:
                Amatrix[i, i - 1] = self.lambdas[ASK_INDEX] * np.exp(-1)
        return Amatrix, z_vector
    
    def calculate_true_value_function(self, state: np.ndarray):
        current_time = state[0, TIME_INDEX]
        inventories = state[:, INVENTORY_INDEX]
        value_fct = np.zeros(shape=(self.num_trajectories, 1))
        h_t = self._calculate_ht(current_time)
        indices = np.clip(self.max_inventory + inventories, 0, 2 * self.max_inventory)
        indices = indices.astype(int)
        h_0 = h_t[indices]
        value_fct = h_0 + state[:, CASH_INDEX] + state[:, INVENTORY_INDEX] * state[:, ASSET_PRICE_INDEX]
        return value_fct
    
    def _compute_integral(self):
        c = self.gamma * self.sigma * self.fads_proportion
        u0 = 0
        xi=1
        if self.fads_proportion == 0:
            return self.terminal_time
        def m(t):
            return u0 * np.exp(-self.eta * t)

        def v(t):
            return (xi**2) / (2.0 * self.eta) * (1.0 - np.exp(-2.0 * self.eta * t))

        def integrand(t):
            return np.exp(c * m(t) + 0.5 * (c**2) * v(t))
        integral, _ = quad(integrand, 0.0, self.terminal_time)
        return integral


class CarteaJaimungalOeAgent(Agent): # optimal execution agent (liquidation problem)
    def __init__(
        self,
        phi: float = 2 * 10 ** (-4),
        alpha: float = 0.0001,
        env: TradingEnvironment = None,
    ):
        self.phi = phi
        self.alpha = alpha
        self.env = env or TradingEnvironment()
        self.price_impact_model = env.model_dynamics.price_impact_model
        assert isinstance(self.env.model_dynamics, TradinghWithSpeedModelDynamics), "Trader must be type TradinghWithSpeedTrader"
        self.terminal_time = self.env.terminal_time
        self.temporary_price_impact = self.price_impact_model.temporary_impact_coefficient
        self.permanent_price_impact = self.price_impact_model.permanent_impact_coefficient
        self.num_trajectories = self.env.num_trajectories

    def get_action(self, state: np.ndarray):
        action = np.zeros(shape=(self.num_trajectories, 1))
        # The formulae below is in page 147 of Cartea, Jaimungal, Penalva (2015)
        # Algorithmic and High-Frequency Trading
        # Cambridge University Press
        gamma = np.sqrt(self.phi / self.temporary_price_impact)
        zeta = (self.alpha - 0.5 * self.permanent_price_impact + np.sqrt(self.temporary_price_impact * self.phi)) / (
            self.alpha - 0.5 * self.permanent_price_impact - np.sqrt(self.temporary_price_impact * self.phi)
        )
        initial_inventory = self.env.initial_inventory

        time_left = self.terminal_time - state[0, TIME_INDEX]
        action[:, :] = (
            gamma
            * initial_inventory
            * (
                (zeta * np.exp(gamma * time_left) + np.exp(-gamma * time_left))
                / (zeta * np.exp(gamma * self.terminal_time) - np.exp(-gamma * self.terminal_time))
            )
        )
        return -np.sign(initial_inventory) * action


class MMwithFadsInformedUniformedTradersAgent(Agent):
    def __init__(self, 
                 env: TradingEnvironment = None):
        self.env = env or TradingEnvironment()
        # Store all parameters
        self.step_size = self.env.model_dynamics.midprice_model.step_size
        self.big_phi = self.env.reward_function.per_step_inventory_aversion
        self.alpha = self.env.reward_function.terminal_inventory_aversion
        self.mu = self.env.model_dynamics.midprice_model.drift
        self.eta = self.env.model_dynamics.midprice_model.eta
        self.gamma = self.env.model_dynamics.arrival_model.gamma
        self.phi = self.env.model_dynamics.arrival_model.phi
        self.psi = self.env.model_dynamics.arrival_model.psi
        self.k = self.env.model_dynamics.fill_probability_model.fill_exponent
        self.sigma = self.env.model_dynamics.midprice_model.volatility
        self.fads_proportion =  self.env.model_dynamics.midprice_model.fads_proportion

        # self.risk_aversion = risk_aversion              # Risk aversion parameter ??? needed maybe just for making comparison with Avellaneda-Stoikov Agent
        self.env = env or TradingEnvironment()
        assert isinstance(self.env, TradingEnvironment)
        self.terminal_time = self.env.terminal_time
        self.volatility = self.env.model_dynamics.midprice_model.volatility 
        self.fill_exponent = self.env.model_dynamics.fill_probability_model.fill_exponent

    def get_action(self, state: np.ndarray):
        #inventory = state[:, INVENTORY_INDEX]
        time = state[:, TIME_INDEX]
        action = self._get_action(time, state)
        if action.min() < 0:
            warnings.warn("MM agent is quoting a negative spread")
        return action

    def _get_spreads(self, time: float, state: np.ndarray) -> float:
        values_functions = self._approximate_value_functions(state, inventories_add=[-1, 0, 1])
        value_function_q_neg = values_functions[-1.0]
        value_function_q = values_functions[0.0]
        value_function_q_pos = values_functions[1.0]
        ask_spread = 1/self.k - value_function_q_neg + value_function_q
        bid_spread = 1/self.k - value_function_q_pos + value_function_q
        #print("Bid spread:", bid_spread, "Ask spread:", ask_spread)
        return [bid_spread, ask_spread]
    
    def _get_action(self, time: float, state: np.ndarray):
        bid_half_spread, ask_half_spread = self._get_spreads(time, state)
        # Convert to column vectors and concatenate horizontally
        bid_half_spread = bid_half_spread.reshape(-1, 1)
        ask_half_spread = ask_half_spread.reshape(-1, 1)
        return np.concatenate([bid_half_spread, ask_half_spread], axis=1)
    
    def _approximate_value_functions(self, state: np.ndarray, inventories_add=[-1, 0, 1]):
        """
        Approximates the value function for multiple inventory levels.
        
        Parameters
        ----------
        state : np.ndarray
            Current state containing inventory, time, and market info.
        inventories : list of floats
            List of inventory levels q at which to evaluate the value function.

        Returns
        -------
        values : dict
            Dictionary mapping q -> V(t,q).
        """
        inventories = state[:, INVENTORY_INDEX]
        time = state[:, TIME_INDEX]
        market_state = state[:, FADS_INDEX]

        # Extract scalar time since all trajectories have the same time
        current_time = time[0]  # Use first element since all are the same

        # Compute coefficients
        A = self._comp_A_explicit(current_time)
        b0, b1, c0, c1, c2 = self._solve_BC_system(current_time)

        values = {}
        for q in inventories_add:
            inventory = inventories + q
            B = (b0 + market_state * b1)
            C = (c0 + market_state * c1 + market_state**2 * c2)
            V = (inventory**2 * A + inventory * B + C)
            values[q] = V

        return values
        
    # auxiliary functions for the finding approximate value function
    def _comp_A_explicit(self, t):
        """
        Computes the explicit function A(t) based on the parameters big_phi, phi, psi, k, and alpha.
        """

        # Useful quantities
        sqrt_big_phi = np.sqrt(self.big_phi)
        call_kappa = 4 * (self.phi + self.psi) * np.exp(-1) * self.k
        sqrt_call_kappa = np.sqrt(call_kappa)
        beta = (sqrt_big_phi + sqrt_call_kappa * self.alpha)/ (sqrt_big_phi - sqrt_call_kappa * self.alpha)

        if np.isclose(sqrt_big_phi, sqrt_call_kappa * self.alpha):
            return -self.alpha
        else:
            exp_term = np.exp(2 * sqrt_big_phi * sqrt_call_kappa * (self.terminal_time - t))
            numerator = sqrt_big_phi * (1 - exp_term * beta)
            denominator = sqrt_call_kappa * (1 + exp_term * beta)
            return numerator / denominator
        


    def _solve_BC_system(self, t):
        """
        Solve the coupled system for B(t) and C(t) backward from T to 0.
        Returns time grid and solutions for b0, b1, c0, c1, c2.
        """

        def rhs(t, y):
            # y = [b0, b1, c0, c1, c2]
            b0, b1, c0, c1, c2 = y

            # --- compute B-part ---
            db0, db1 = self._compute_B_rhs(t, [b0, b1])

            # --- compute C-part (depends on b0, b1) ---
            dc0, dc1, dc2 = self._compute_C_rhs(t, [c0, c1, c2], b0, b1)

            return [db0, db1, dc0, dc1, dc2]

        # Terminal conditions at t = T
        yT = [0.0, 0.0, 0.0, 0.0, 0.0]

        # Integrate backward from T -> 0
        sol = solve_ivp(rhs, [self.terminal_time, 0.0], yT,
                        method="RK45", dense_output=True)

        y_at_t = sol.sol(t)
        # print("y_at_t", y_at_t  )
        b0, b1, c0, c1, c2 = y_at_t
        return b0, b1, c0, c1, c2

    
    def _compute_B_rhs(self, t, B):
        """
        Computes the right-hand side of the ODE for B(t).
        Here t is backward time (integrating from T to 0).
        """
        A = self._comp_A_explicit(t)
        b0, b1 = B

        db0 = -self.mu - 4 * self.k * (self.psi + self.phi) * np.exp(-1) * A * b0 # if mu is =0, -> b0 = 0 for every t (proved by printing db0)
        db1 = (+ self.eta * self.sigma * self.fads_proportion + self.eta * b1
                - 4 * (self.psi + self.phi) * np.exp(-1) * A * self.k * b1 
                - 4 * np.exp(-1) * self.psi * self.fads_proportion * self.sigma * self.gamma * A 
                - 4 * np.exp(-1) * self.k * self.gamma * self.fads_proportion * self.sigma * self.psi * A**2
                )
        return [db0, db1]


    def _compute_C_rhs(self, t, C, b0, b1):
        """
        Computes the right-hand side of the ODE for C(t).
        Here t is backward time (integrating from T to 0).
        """
        A = self._comp_A_explicit(t)
        c0, c1, c2 = C
        
        # Equation (38), lines 3–5
        dc0 = (- c2 - (1 / self.k) * np.exp(-1) * 
               (2 * (self.psi + self.phi) 
            + 2 * self.k * A * (self.phi + self.psi) 
            + self.k**2 * (self.phi + self.psi) * (A**2 + b0**2))
                )
        dc1 = + self.eta * c1 - (1 / self.k) * np.exp(-1) * (
            2 * self.psi * self.k * self.sigma * self.gamma * self.fads_proportion * b0 
            + self.k**2 * (self.phi + self.psi) * (2 * b0 * b1) +
            2 * self.k**2 * self.psi * self.sigma * self.gamma * self.fads_proportion * A * b0
        )
        dc2 = 2 * self.eta * c2 - (1 / self.k) * np.exp(-1) * (
            self.k**2 * (self.phi + self.psi) * b1**2 
            + 2 * self.k**2 * self.psi * self.sigma * self.gamma * self.fads_proportion * A * b1
            + 2 * self.psi * self.k * self.sigma * self.gamma * self.fads_proportion * b1
        )
        
        return [dc0, dc1, dc2]



class OptimizedFullInfoMMwithFadsInformedUniformedTradersAgent(Agent):
    def __init__(self, 
                 env: TradingEnvironment = None,):
        
        self.env = env or TradingEnvironment()
        # Store all parameters
        self.step_size = self.env.model_dynamics.midprice_model.step_size
        self.big_phi = self.env.reward_function.per_step_inventory_aversion
        self.alpha = self.env.reward_function.terminal_inventory_aversion
        self.mu = self.env.model_dynamics.midprice_model.drift
        self.eta = self.env.model_dynamics.midprice_model.eta
        self.gamma = self.env.model_dynamics.arrival_model.gamma
        self.phi = self.env.model_dynamics.arrival_model.phi
        self.psi = self.env.model_dynamics.arrival_model.psi
        # TODO attention here, what is the fill exponent if we have FadsInformedUniformedTradersArrivalModel?
        self.k = self.env.model_dynamics.fill_probability_model.fill_exponent
        #self.k = self.env.model_dynamics.arrival_model.k
        self.sigma = self.env.model_dynamics.midprice_model.volatility
        self.fads_proportion =  self.env.model_dynamics.midprice_model.fads_proportion
        # self.risk_aversion = risk_aversion       
        assert isinstance(self.env, TradingEnvironment)
        self.terminal_time = self.env.terminal_time
        self.volatility = self.env.model_dynamics.midprice_model.volatility
        
        # Pre-compute all ODE solutions at initialization
        self.time_grid_size = int(self.terminal_time // self.step_size + 1)
        #self._print_parameters()
        self._precompute_all_solutions()
        print(f"✓ Pre-computed ODE solutions for {self.time_grid_size} time points")
    
    def _precompute_all_solutions(self):
        """
        Solve the ODE system once for the entire time horizon.
        Store results for fast lookup during trading.
        """
        def rhs(t, y):
            # y = [b0, b1, c0, c1, c2]
            b0, b1, c0, c1, c2 = y
            db0, db1 = self._compute_B_rhs(t, [b0, b1])
            dc0, dc1, dc2 = self._compute_C_rhs(t, [c0, c1, c2], b0, b1)
            return [db0, db1, dc0, dc1, dc2]
        
        # Terminal conditions at t = T
        yT = [0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Create time grid from T to 0
        self.time_grid = np.linspace(self.terminal_time, 0.0, self.time_grid_size)
        
        sol = solve_ivp(rhs, [self.terminal_time, 0.0], yT,
                       t_eval=self.time_grid, method="BDF", rtol=1e-6)
               
        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")
        
        # Store solutions for fast lookup
        # sol.y has shape (5, time_grid_size)
        self.precomputed_b0 = sol.y[0, :]  # b0 for all times
        self.precomputed_b1 = sol.y[1, :]  # b1 for all times
        self.precomputed_c0 = sol.y[2, :]  # c0 for all times
        self.precomputed_c1 = sol.y[3, :]  # c1 for all times
        self.precomputed_c2 = sol.y[4, :]  # c2 for all times
        
        # Also precompute A(t) for all times (since it's analytical)
        self.precomputed_A = np.array([self._comp_A_explicit(t) for t in self.time_grid])
        # print("Precomputed A(T) value:", self.precomputed_A[0])
        # print("Precomputed A(0) value:", self.precomputed_A[-1])
    
    def _get_coefficients_fast(self, t: float):
        """
        Fast lookup of ODE solutions at time t using interpolation.
        This replaces the expensive _solve_BC_system call.
        """
        # Handle boundary cases
        t = np.clip(t, 0.0, self.terminal_time)
        
        # Linear interpolation
        b0 = np.interp(t, self.time_grid, self.precomputed_b0)
        b1 = np.interp(t, self.time_grid, self.precomputed_b1)
        c0 = np.interp(t, self.time_grid, self.precomputed_c0)
        c1 = np.interp(t, self.time_grid, self.precomputed_c1)
        c2 = np.interp(t, self.time_grid, self.precomputed_c2)
        A = np.interp(t, self.time_grid, self.precomputed_A)
        
        return A, b0, b1, c0, c1, c2
    
    def get_action(self, state: np.ndarray):
        time = state[:, TIME_INDEX]
        action = self._get_action(time, state)
        if action.min() < 0:
            warnings.warn("MM agent is quoting a negative spread")
        return action

    def _get_spreads(self, time: float, state: np.ndarray) -> tuple:
        # state is expected to be a 2D array, shape (1, n_features)
        inventory = state[:, INVENTORY_INDEX]
        time = state[:, TIME_INDEX]
        market_state = state[:, FADS_INDEX]
        
        current_time = time[0]  # Use first element since all are the same

        # Get coefficients for the current time
        A, b0, b1, c0, c1, c2 = self._get_coefficients_fast(current_time)
        
        # Calculate B(t,u)
        B = b0 + market_state * b1
        
        # Use the direct analytical formula for spreads
        ask_spread = (1 / self.k) + (2 * inventory - 1) * A + B
        bid_spread = (1 / self.k) - (2 * inventory + 1) * A - B
        
        return bid_spread, ask_spread
        
    def _get_action(self, time: float, state: np.ndarray):
        bid_half_spread, ask_half_spread = self._get_spreads(time, state)
        bid_half_spread = bid_half_spread.reshape(-1, 1)
        ask_half_spread = ask_half_spread.reshape(-1, 1)
        return np.concatenate([bid_half_spread, ask_half_spread], axis=1)
    
    # Auxiliary functions but
    def _comp_A_explicit(self, t):
        sqrt_big_phi = np.sqrt(self.big_phi)
        call_kappa = 4 * (self.phi + self.psi) * np.exp(-1) * self.k
        sqrt_call_kappa = np.sqrt(call_kappa)
        beta = (sqrt_big_phi + sqrt_call_kappa * self.alpha)/ (sqrt_big_phi - sqrt_call_kappa * self.alpha)

        if np.isclose(sqrt_big_phi, sqrt_call_kappa * self.alpha):
            return -self.alpha
        else:
            #exp_term = np.exp(2 * sqrt_big_phi * call_kappa * (self.terminal_time - t))
            exp_term = np.exp(2 * sqrt_big_phi * sqrt_call_kappa * (self.terminal_time - t))
            numerator = sqrt_big_phi * (1 - exp_term * beta)
            denominator = sqrt_call_kappa * (1 + exp_term * beta)
            return numerator / denominator
    
    def _compute_B_rhs(self, t, B):
        A = self._comp_A_explicit(t)
        b0, b1 = B

        db0 = -self.mu - 4 * self.k * (self.psi + self.phi) * np.exp(-1) * A * b0
        db1 = (+ self.eta * self.sigma * self.fads_proportion + self.eta * b1
                - 4 * (self.psi + self.phi) * np.exp(-1) * A * self.k * b1 
                - 4 * np.exp(-1) * self.psi * self.fads_proportion * self.sigma * self.gamma * A 
                - 4 * np.exp(-1) * self.k * self.gamma * self.fads_proportion * self.sigma * self.psi * A**2
                )
        return [db0, db1]

    def _compute_C_rhs(self, t, C, b0, b1):
        A = self._comp_A_explicit(t)
        c0, c1, c2 = C
        
        dc0 = (- c2 - (np.exp(-1) / self.k) * 
           (2 * (self.psi + self.phi) 
        + 2 * self.k * A * (self.phi + self.psi) 
        + self.k**2 * (self.phi + self.psi) * (A**2 + b0**2))
            )
        dc1 = + self.eta * c1 - (np.exp(-1) / self.k) *  (
            2 * self.psi * self.k * self.sigma * self.gamma * self.fads_proportion * b0 
            + self.k**2 * (self.phi + self.psi) * (2 * b0 * b1) +
            2 * self.k**2 * self.psi * self.sigma * self.gamma * self.fads_proportion * A * b0
        )
        dc2 = 2 * self.eta * c2 - (np.exp(-1) / self.k) * (
            self.k**2 * (self.phi + self.psi) * b1**2 
            + 2 * self.k**2 * self.psi * self.sigma * self.gamma * self.fads_proportion * A * b1
            + 2 * self.psi * self.k * self.sigma * self.gamma * self.fads_proportion * b1
        )
        
        return [dc0, dc1, dc2]

    def plot_precomputed_solutions(self):
        """Visualizes the precomputed ODE solutions."""
        fig, axs = plt.subplots(3, 2, figsize=(12, 12), sharex=True)
        fig.suptitle("Precomputed ODE Coefficients vs. Time-to-Maturity")
        
        # Time runs from T to 0, so we plot against T-t
        time_to_maturity = self.terminal_time - self.time_grid
        
        axs[0, 0].plot(time_to_maturity, self.precomputed_A)
        axs[0, 0].set_title("A(t)")
        
        axs[0, 1].plot(time_to_maturity, self.precomputed_b0)
        axs[0, 1].set_title("b0(t)")
        
        axs[1, 0].plot(time_to_maturity, self.precomputed_b1)
        axs[1, 0].set_title("b1(t)")
        
        axs[1, 1].plot(time_to_maturity, self.precomputed_c0)
        axs[1, 1].set_title("c0(t)")
        
        axs[2, 0].plot(time_to_maturity, self.precomputed_c1)
        axs[2, 0].set_title("c1(t)")
        
        axs[2, 1].plot(time_to_maturity, self.precomputed_c2)
        axs[2, 1].set_title("c2(t)")
        
        for ax in axs.flat:
            ax.set_xlabel("Time to Maturity (T-t)")
            ax.grid(True)
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()





# class MMwithFadsInformedUniformedTradersAgentExact(Agent):
#     def __init__(self, 
#                  mu: float = 0, 
#                  env: TradingEnvironment = None):
#         self.big_phi = big_phi
#         self.alpha = alpha
#         self.mu = mu
#         self.eta = eta
#         self.gamma = gamma
#         self.phi = phi
#         self.psi = psi
#         self.k = k
#         self.sigma = sigma
#         self.fads_proportion = fads_proportion
#         #self.risk_aversion = risk_aversion              # Risk aversion parameter ??? needed maybe just for making comparison with Avellaneda-Stoikov Agent
#         self.env = env or TradingEnvironment()
#         assert isinstance(self.env, TradingEnvironment)
#         self.terminal_time = self.env.terminal_time
#         self.volatility = self.env.model_dynamics.midprice_model.volatility # maybe is our sigma?
#         self.fill_exponent = self.env.model_dynamics.fill_probability_model.fill_exponent

#     def get_action(self, state: np.ndarray):
#         #inventory = state[:, INVENTORY_INDEX]
#         time = state[:, TIME_INDEX]
#         action = self._get_action(time, state)
#         if action.min() < 0:
#             warnings.warn("MM agent is quoting a negative spread")
#         return action

#     def _get_spreads(self, time: float, state: np.ndarray) -> float:
#         values_functions = self._approximate_value_functions(state, inventories_add=[-1, 0, 1])
#         value_function_q_neg = values_functions[-1.0]
#         value_function_q = values_functions[0.0]
#         value_function_q_pos = values_functions[1.0]
#         ask_spread = 1/self.k - value_function_q_neg + value_function_q
#         bid_spread = 1/self.k - value_function_q_pos + value_function_q
#         #print("Bid spread:", bid_spread, "Ask spread:", ask_spread)
#         return [bid_spread, ask_spread]
    
#     def _get_action(self, time: float, state: np.ndarray):
#         bid_half_spread, ask_half_spread = self._get_spreads(time, state)
#         # Convert to column vectors and concatenate horizontally
#         bid_half_spread = bid_half_spread.reshape(-1, 1)
#         ask_half_spread = ask_half_spread.reshape(-1, 1)
#         return np.concatenate([bid_half_spread, ask_half_spread], axis=1)
    
#     def _approximate_value_functions(self, state: np.ndarray, inventories_add=[-1, 0, 1]):
#         """
#         Approximates the value function for multiple inventory levels.
        
#         Parameters
#         ----------
#         state : np.ndarray
#             Current state containing inventory, time, and market info.
#         inventories : list of floats
#             List of inventory levels q at which to evaluate the value function.

#         Returns
#         -------
#         values : dict
#             Dictionary mapping q -> V(t,q).
#         """
#         inventories = state[:, INVENTORY_INDEX]
#         time = state[:, TIME_INDEX]
#         print("Time:", time)
#         market_state = state[:, FADS_INDEX] 

#         # Extract scalar time since all trajectories have the same time
#         current_time = time[0]  # Use first element since all are the same

#         # Compute coefficients
#         A = self._comp_A_explicit(current_time)
#         b0, b1, c0, c1, c2 = self._solve_BC_system(current_time)

#         values = {}
#         for q in inventories_add:
#             inventory = inventories + q
#             B = (b0 + market_state * b1)
#             C = (c0 + market_state * c1 + market_state**2 * c2)
#             V = (inventory**2 * A + inventory * B + C)
#             values[q] = V

#         return values
        
#     # auxiliary functions for the finding approximate value function
#     def _comp_A_explicit(self, t):
#         """
#         Computes the explicit function A(t) based on the parameters big_phi, phi, psi, k, and alpha.
#         """

#         # Useful quantities
#         sqrt_big_phi = np.sqrt(self.big_phi)
#         call_kappa = 4 * (self.phi + self.psi) * np.exp(-1) * self.k
#         sqrt_call_kappa = np.sqrt(call_kappa)
#         beta = (sqrt_big_phi + sqrt_call_kappa * self.alpha)/ (sqrt_big_phi - sqrt_call_kappa * self.alpha)

#         if np.isclose(sqrt_big_phi, sqrt_call_kappa * self.alpha):
#             return - self.alpha
#         else:
#             exp_term = np.exp(2 * sqrt_big_phi * sqrt_call_kappa * (self.terminal_time - t))
#             numerator = sqrt_big_phi * (1 - exp_term * beta)
#             denominator = sqrt_call_kappa * (1 + exp_term * beta)
#             return numerator / denominator
        
#     def solve_b0(self, t):
#         """
#         Solve the ODE for b0(t)
#         Returns time grid and solution for b0.
#         """
#         constant_t = - 4 * self.k * (self.psi + self.phi) * np.exp(-1) * self._comp_A_explicit(t)
#         #b0_t = np.exp(constant_t * t) + integrate.quad(lambda x: self.mu * np.exp(-constant_t * (t-x) ), 0, t)
    
#     def solve_b0(self, T, n_grid=1000):
#         t_grid = np.linspace(0, T, n_grid)
#         A_vals = self._comp_A_explicit(t_grid)
#         lam_vals = 4 * self.k * (self.psi + self.phi) * np.exp(-1) * A_vals
    
#         # Compute integral of lambda up to each t
#         int_lam = np.cumsum(lam_vals) * (T / (n_grid-1))  # trapezoid approx
    
#         I_vals = np.exp(int_lam)

#         b0_vals = np.zeros_like(t_grid)
#         for i, t in enumerate(t_grid):
#             # integral of I(s) from 0 to t
#             int_I = np.trapz(I_vals[:i+1], t_grid[:i+1])
#             b0_vals[i] = -self.mu * (1 / I_vals[i]) * int_I
        
#         return t_grid, b0_vals

#     def _solve_BC_system(self, t):
#         """
#         Solve the coupled system for B(t) and C(t) backward from T to 0.
#         Returns time grid and solutions for b0, b1, c0, c1, c2.
#         """

#         def rhs(t, y):
#             # y = [b0, b1, c0, c1, c2]
#             b0, b1, c0, c1, c2 = y

#             # --- compute B-part ---
#             db0, db1 = self._compute_B_rhs(t, [b0, b1])

#             # --- compute C-part (depends on b0, b1) ---
#             dc0, dc1, dc2 = self._compute_C_rhs(t, [c0, c1, c2], b0, b1)

#             return [db0, db1, dc0, dc1, dc2]

#         # Terminal conditions at t = T
#         yT = [0.0, 0.0, 0.0, 0.0, 0.0]

#         # Integrate backward from T -> 0
#         sol = solve_ivp(rhs, [self.terminal_time, 0.0], yT,
#                         method="RK45", dense_output=True)

#         # Evaluate the solution at time t
#         y_at_t = sol.sol(t)
#         b0, b1, c0, c1, c2 = y_at_t
#         return b0, b1, c0, c1, c2

    
#     def _compute_B_rhs(self, t, B):
#         """
#         Computes the right-hand side of the ODE for B(t).
#         Here t is backward time (integrating from T to 0).
#         """
#         A = self._comp_A_explicit(t)
#         b0, b1 = B

#         db0 = -self.mu - 4 * self.k * (self.psi + self.phi) * np.exp(-1) * A * b0 # if mu is =0, -> b0 = 0 for every t
#         db1 = + self.eta * self.sigma * self.fads_proportion + self.eta * b1  - 4 * (self.psi + self.phi) * np.exp(-1) * A * b1 - 4 * np.exp(-1) * self.psi * self.fads_proportion * self.sigma * self.gamma * A - 4 * np.exp(-1) * self.k * self.gamma * self.fads_proportion * self.sigma * self.psi * A**2    
#         return [db0, db1]


#     def _compute_C_rhs(self, t, C, b0, b1):
#         """
#         Computes the right-hand side of the ODE for C(t).
#         Here t is backward time (integrating from T to 0).
#         """
#         A = self._comp_A_explicit(t)
#         c0, c1, c2 = C
        
#         # Equation (38), lines 3–5
#         dc0 = - c2 - (1 / self.k) * np.exp(-1) * (2 * (self.psi + self.phi) + 2 * self.k * A * (self.phi + self.psi) + self.k**2 * (self.phi + self.psi) * (A**2 + b0**2))
#         dc1 = + self.eta * c1 - (1 / self.k) * np.exp(-1) * (
#             2 * self.psi * self.k * self.sigma * self.gamma * self.fads_proportion * b0 +
#             self.k**2 * (self.phi + self.psi) * (2 * b0 * b1) +
#             2 * self.k**2 * self.psi * self.sigma * self.gamma * self.fads_proportion * A * b0
#         )
#         dc2 = 2 * self.eta * c2 - (1 / self.k) * np.exp(-1) * (
#             self.k**2 * (self.phi + self.psi) * b1**2 +
#             2 * self.k**2 * self.psi * self.sigma * self.gamma * self.fads_proportion * A * b1 +
#             2 * self.psi * self.k * self.sigma * self.gamma * self.fads_proportion * b1
#         )
        
#         return [dc0, dc1, dc2]





class OptimizedPartialInfoMMwithFadsInformedUniformedTradersAgent(Agent):
    def __init__(self,  
                 env: TradingEnvironment = None,):
        
        self.env = env or TradingEnvironment()
        # Store all parameters
        self.step_size = self.env.model_dynamics.midprice_model.step_size
        self.big_phi = self.env.reward_function.per_step_inventory_aversion
        self.alpha = self.env.reward_function.terminal_inventory_aversion
        self.mu = self.env.model_dynamics.midprice_model.drift
        self.eta = self.env.model_dynamics.midprice_model.eta
        self.gamma = self.env.model_dynamics.arrival_model.gamma
        self.phi = self.env.model_dynamics.arrival_model.phi
        self.psi = self.env.model_dynamics.arrival_model.psi
        # TODO attention here, what is the fill exponent if we have FadsInformedUniformedTradersArrivalModel?
        #self.k = self.env.model_dynamics.arrival_model.k
        self.k = self.env.model_dynamics.fill_probability_model.fill_exponent
        self.sigma = self.env.model_dynamics.midprice_model.volatility
        self.fads_proportion =  self.env.model_dynamics.midprice_model.fads_proportion
        assert isinstance(self.env, TradingEnvironment)
        self.terminal_time = self.env.terminal_time
        self.volatility = self.env.model_dynamics.midprice_model.volatility
        self.time_grid_size = int(self.terminal_time // self.step_size + 1)

        self._precompute_all_solutions()
        print(f"✓ Pre-computed ODE solutions for {self.time_grid_size} time points")
    
    def _precompute_all_solutions(self):
        """
        Solve the ODE system once for the entire time horizon.
        Store results for fast lookup during trading.
        """

        def rhs(t, y):
            b0, b1, c0, c1, c2 = y
            db0, db1 = self._compute_B_rhs(t, [b0, b1])
            dc0, dc1, dc2 = self._compute_C_rhs(t, [c0, c1, c2], b0, b1)
            out = [db0, db1, dc0, dc1, dc2]
            if any(np.isnan(out)):
                raise ValueError(f"NaN in RHS at t={t}, y={y}")
            # import sys
            # print(f"rhs call at t={t:.4f}, y={y} → {out}")
            # sys.stdout.flush()
            return out
        
        # Terminal conditions at t = T
        yT = [0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Create time grid from T to 0
        self.time_grid = np.linspace(self.terminal_time, 0.0, self.time_grid_size)
        
        # Solve ODE once for all time points
        sol = solve_ivp(rhs, [self.terminal_time, 0.0], yT,
                       t_eval=self.time_grid, method="BDF", rtol=1e-6)
        
        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")
        
        # Store solutions for fast lookup
        # sol.y has shape (5, time_grid_size)
        self.precomputed_b0 = sol.y[0, :]  # b0 for all times
        self.precomputed_b1 = sol.y[1, :]  # b1 for all times
        self.precomputed_c0 = sol.y[2, :]  # c0 for all times
        self.precomputed_c1 = sol.y[3, :]  # c1 for all times
        self.precomputed_c2 = sol.y[4, :]  # c2 for all times
        
        # Also precompute A(t) for all times (since it's analytical)
        self.precomputed_A = np.array([self._comp_A_explicit(t) for t in self.time_grid])
        
    def _get_coefficients_fast(self, t: float):
        """
        Fast lookup of ODE solutions at time t using interpolation.
        """
        # Handle boundary cases
        t = np.clip(t, 0.0, self.terminal_time)
        
        # Linear interpolation (could use higher-order if needed)
        b0 = np.interp(t, self.time_grid, self.precomputed_b0)
        b1 = np.interp(t, self.time_grid, self.precomputed_b1)
        c0 = np.interp(t, self.time_grid, self.precomputed_c0)
        c1 = np.interp(t, self.time_grid, self.precomputed_c1)
        c2 = np.interp(t, self.time_grid, self.precomputed_c2)
        A = np.interp(t, self.time_grid, self.precomputed_A)
        
        return A, b0, b1, c0, c1, c2
    
    def get_action(self, state: np.ndarray):
        time = state[:, TIME_INDEX]
        action = self._get_action(time, state)
        if action.min() < 0:
            warnings.warn("MM agent is quoting a negative spread")
        return action

    def _get_spreads(self, time: float, state: np.ndarray) -> float:
        values_functions = self._approximate_value_functions(state, inventories_add=[-1, 0, 1])
        value_function_q_neg = values_functions[-1.0]
        value_function_q = values_functions[0.0]
        value_function_q_pos = values_functions[1.0]
        ask_spread = 1/self.k - value_function_q_neg + value_function_q
        bid_spread = 1/self.k - value_function_q_pos + value_function_q
        return [bid_spread, ask_spread]
    
    def _get_action(self, time: float, state: np.ndarray):
        bid_half_spread, ask_half_spread = self._get_spreads(time, state)
        bid_half_spread = bid_half_spread.reshape(-1, 1)
        ask_half_spread = ask_half_spread.reshape(-1, 1)
        return np.concatenate([bid_half_spread, ask_half_spread], axis=1)
    
    def _approximate_value_functions(self, state: np.ndarray, inventories_add=[-1, 0, 1]):

        inventories = state[:, INVENTORY_INDEX]
        time = state[:, TIME_INDEX]
        market_state = state[:, FILTERED_FADS_INDEX]
        
        current_time = time[0]  # Use first element since all are the same
        
        # FAST: Get coefficients via interpolation (not ODE solving!)
        A, b0, b1, c0, c1, c2 = self._get_coefficients_fast(current_time)
        
        values = {}
        for q in inventories_add:
            inventory = inventories + q
            B = (b0 + market_state * b1)
            C = (c0 + market_state * c1 + market_state**2 * c2)
            V = (-inventory**2 * A - inventory * B + C)
            values[q] = V
        
        return values

    # Auxiliary functions
    def _comp_A_explicit(self, t):
        sqrt_big_phi = np.sqrt(self.big_phi)
        call_kappa = 4 * (self.phi + self.psi) * np.exp(-1) * self.k
        sqrt_call_kappa = np.sqrt(call_kappa)
        beta = (sqrt_big_phi + sqrt_call_kappa * self.alpha)/ (sqrt_big_phi - sqrt_call_kappa * self.alpha)

        if np.isclose(sqrt_big_phi, sqrt_call_kappa * self.alpha):
            return +self.alpha
        else:
            exp_term = np.exp(2 * sqrt_big_phi * sqrt_call_kappa * (self.terminal_time - t))
            numerator = sqrt_big_phi * (1 - exp_term * beta)
            denominator = sqrt_call_kappa * (1 + exp_term * beta)
            return -numerator / denominator

    def _compute_B_rhs(self, t, B):
        """
        b0' = mu + 4 e^{-1} (phi+psi) k b0 A
        b1' = eta b1 - eta sigma q + psi e^{-1} (-4 A gamma sigma q + 4 k q gamma sigma A^2)
              + 4 e^{-1} (phi+psi) k b1 A
        """
        A = self._comp_A_explicit(t)
        b0, b1 = B

        q = self.fads_proportion   # q in the paper
        expm1 = np.exp(-1.0)
        common = 4.0 * expm1 * (self.phi + self.psi) * self.k

        db0 = - self.mu - common * b0 * A

        # term from psi e^{-1} ( -4 A gamma sigma q + 4 k q gamma sigma A^2 )
        psi_term = self.psi *  expm1 * (
            -4.0 * A * self.gamma * self.sigma * q
            + 4.0 * self.k * q * self.gamma * self.sigma * (A**2)
        )

        db1 = self.eta * b1 - self.eta * self.sigma * q + psi_term + common * b1 * A

        return [db0, db1]

    def _compute_C_rhs(self, t, C, b0, b1):
        """
        c0' = -sigma^2 c2 + e^{-1}/k (phi+psi) (2 - 2 k A + k^2 A^2 + k^2 b0^2)
        c1' = eta c1 - 2 e^{-1} psi b0 q gamma sigma + 2 e^{-1} psi k gamma sigma q b0 A
               + 2 e^{-1} k (phi+psi) b0 b1
        c2' = 2 eta c2 - 2 e^{-1} psi b1 q gamma sigma + 2 e^{-1} psi k gamma sigma q A b1
               + e^{-1} k (phi+psi) b1^2
        """
        A = self._comp_A_explicit(t)
        c0, c1, c2 = C
        q = self.fads_proportion
        expm1 = np.exp(-1.0)

        dc0 = (- (self.sigma**2) * c2
               + (expm1 / self.k) * (self.phi + self.psi) * (2.0 - 2.0 * self.k * A + (self.k**2) * (A**2) + (self.k**2) * (b0**2))
               )

        dc1 = (self.eta * c1
               - 2.0 * expm1 * self.psi * b0 * q * self.gamma * self.sigma
               + 2.0 * expm1 * self.psi * self.k * self.gamma * self.sigma * q * b0 * A
               + 2.0 * expm1 * self.k * (self.phi + self.psi) * b0 * b1
               )

        dc2 = (2.0 * self.eta * c2
               - 2.0 * expm1 * self.psi * b1 * q * self.gamma * self.sigma
               + 2.0 * expm1 * self.psi * self.k * self.gamma * self.sigma * q * A * b1
               + expm1 * self.k * (self.phi + self.psi) * (b1**2)
               )

        return [dc0, dc1, dc2]
    

    def plot_precomputed_solutions(self):
        """Visualizes the precomputed ODE solutions."""
        fig, axs = plt.subplots(3, 2, figsize=(12, 12), sharex=True)
        fig.suptitle("Precomputed ODE Coefficients vs. Time-to-Maturity")
        
        # Time runs from T to 0, so we plot against T-t
        time_to_maturity = self.terminal_time - self.time_grid
        
        axs[0, 0].plot(time_to_maturity, self.precomputed_A)
        axs[0, 0].set_title("A(t)")
        
        axs[0, 1].plot(time_to_maturity, self.precomputed_b0)
        axs[0, 1].set_title("b0(t)")
        
        axs[1, 0].plot(time_to_maturity, self.precomputed_b1)
        axs[1, 0].set_title("b1(t)")
        
        axs[1, 1].plot(time_to_maturity, self.precomputed_c0)
        axs[1, 1].set_title("c0(t)")
        
        axs[2, 0].plot(time_to_maturity, self.precomputed_c1)
        axs[2, 0].set_title("c1(t)")
        
        axs[2, 1].plot(time_to_maturity, self.precomputed_c2)
        axs[2, 1].set_title("c2(t)")
        
        for ax in axs.flat:
            ax.set_xlabel("Time to Maturity (T-t)")
            ax.grid(True)
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
 