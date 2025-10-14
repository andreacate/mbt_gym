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

class CarteaJaimungalMmAgent(Agent): 
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
            self.lambdas = self.env.model_dynamics.arrival_model.intensity
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
            Amatrix[i, i] = -self.phi * self.kappa * inventory**2
            z_term = -self.alpha * self.kappa * inventory**2
            z_vector[i, 0] = np.exp(z_term)
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
            self.gamma = self.env.model_dynamics.arrival_model.gamma
            self.eta = self.env.model_dynamics.midprice_model.eta
            self.phi = self.env.model_dynamics.arrival_model.phi
            self.psi = self.env.model_dynamics.arrival_model.psi
            self.sigma = self.env.model_dynamics.midprice_model.volatility
            self.fads_proportion =  self.env.model_dynamics.midprice_model.fads_proportion
            self.call_kappa = self.phi + self.psi / self.terminal_time * self._compute_integral()
            
            self.lambdas = np.array([[self.call_kappa], [self.call_kappa]])
            # Checked that self.lambdas equals [30, 30] when rescaling psi parameter in psi
            # print("self.lambdas:", self.lambdas)
            self.max_inventory = env.max_inventory
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
        #print("h_t.shape:", h_t.flatten().shape)
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
        #omega_function = np.maximum(omega_function, 1e-16)
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
        self.n_steps = self.env.n_steps
        self.k = self.env.model_dynamics.fill_probability_model.fill_exponent
        self.sigma = self.env.model_dynamics.midprice_model.volatility
        self.fads_proportion =  self.env.model_dynamics.midprice_model.fads_proportion
        assert isinstance(self.env, TradingEnvironment)
        self.terminal_time = self.env.terminal_time
        self.volatility = self.env.model_dynamics.midprice_model.volatility
        
        # Pre-compute all ODE solutions at initialization
        self.time_grid_size = int(self.terminal_time // self.step_size + 1)
        self.time_grid = np.linspace(0.0, self.terminal_time,  self.time_grid_size)

        self._precompute_all_coefficients()

        print(f"✓ Pre-computed ODE solutions for {self.time_grid_size} time points")


    def _precompute_all_coefficients(self):

        self.delta_t = self.terminal_time / (self.n_steps)
        A=np.zeros(self.n_steps)
        b_0 = np.zeros(self.n_steps)
        b_1 = np.zeros(self.n_steps)
        A[-1] = - self.alpha
        c_0 = np.zeros(self.n_steps)
        c_1 = np.zeros(self.n_steps)
        c_2 = np.zeros(self.n_steps)

        self.Gamma = self.gamma * self.sigma * self.fads_proportion

        for t in range(self.n_steps - 2, -1, -1):
            A[t] = A[t+1] + self.delta_t * (- self.big_phi + 4 * (self.psi+ self.phi) * np.exp(-1) * self.k * A[t+1]**2)
            b_0[t] = b_0[t+1] + self.delta_t * (self.mu + 4 *self.k *(self.psi + self.phi) * np.exp(-1) * A[t+1]*b_0[t+1])
            b_1[t] = b_1[t+1] + self.delta_t * (- self.eta * self.sigma * self.fads_proportion - self.eta * b_1[t+1] + 4*(self.psi + self.phi) * np.exp(-1)
                                                *A[t+1] *self.k*b_1[t+1] +4*np.exp(-1)*self.psi * self.Gamma * A[t+1]
                                                +4*np.exp(-1)*self.k*self.Gamma*self.psi*A[t+1]**2)
            c_2[t] = c_2[t+1] + self.delta_t * (- 2*self.eta*c_2[t+1] + (np.exp(-1)/self.k) * (self.k**2*(self.psi + self.phi
                                                )*b_1[t+1]**2 + 2 * self.k**2 * self.psi *self.Gamma * A[t+1]
                                                        *b_1[t+1] + 2*self.psi * self.k*self.Gamma * b_1[t+1]))
            c_1[t] = c_1[t+1] + self.delta_t * (- self.eta * c_1[t+1] + np.exp(-1)/self.k * (2 * self.psi *self.k* self.Gamma
                                                *b_0[t+1] + self.k**2*(self.phi + self.psi) *2*b_0[t+1]*b_1[t+1]
                                                    +2*self.k**2 * self.psi*self.Gamma*b_0[t+1]*A[t+1]))
            c_0[t] = c_0[t+1] + self.delta_t *(c_2[t+1] + (np.exp(-1)/self.k) *( 2*(self.psi + self.phi) + 2*self.k*A[t+1]*(self.phi+self.psi)
                                             + self.k**2 *(self.phi+self.psi)*(A[t+1]**2 + b_0[t+1]**2)))

        self.precomputed_A = A
        self.precomputed_b0 = b_0
        self.precomputed_b1 = b_1
        self.precomputed_c0 = c_0
        self.precomputed_c1 = c_1
        self.precomputed_c2 = c_2
    
    def get_action(self, state: np.ndarray):
        action = self._get_action(state)
        if action.min() < 0:
            warnings.warn("MM agent is quoting a negative spread")
        return action

    
    def _get_spreads(self, state: np.ndarray) -> tuple:
        # state is expected to be a 2D array, shape (1, n_features)
        inventory = state[:, INVENTORY_INDEX]
        time = state[:, TIME_INDEX]
        market_state = state[:, FADS_INDEX]
        
        current_time = time[0]  # Use first element since all are the same

        # Convert current_time to the corresponding index in the precomputed arrays
        # Since time_grid goes from 0 to terminal_time, we can calculate the index directly
        time_index = int(round(current_time / self.step_size))
        time_index = min(time_index, len(self.precomputed_A) - 1)  # Ensure index doesn't exceed array bounds
        
        # Get coefficients directly from precomputed arrays using the time index
        A = self.precomputed_A[time_index]
        b0 = self.precomputed_b0[time_index]
        b1 = self.precomputed_b1[time_index]
        c0 = self.precomputed_c0[time_index]
        c1 = self.precomputed_c1[time_index]
        c2 = self.precomputed_c2[time_index]
        
        # Calculate B(t,u)
        B = b0 + market_state * b1
        
        # Use the direct analytical formula for spreads
        ask_spread = (1 / self.k) + (2 * inventory - 1) * A + B
        bid_spread = (1 / self.k) - (2 * inventory + 1) * A - B
        
        return bid_spread, ask_spread
        
    def _get_action(self, state: np.ndarray):
        bid_half_spread, ask_half_spread = self._get_spreads(state)
        bid_half_spread = bid_half_spread.reshape(-1, 1)
        ask_half_spread = ask_half_spread.reshape(-1, 1)
        return np.concatenate([bid_half_spread, ask_half_spread], axis=1)
          
    #################################################################################
    #  Visualization methods for debugging and analysis

    def plot_precomputed_solutions(self):
        """Visualizes the precomputed ODE solutions."""
        fig, axs = plt.subplots(3, 2, figsize=(12, 12), sharex=True)
        fig.suptitle("Precomputed ODE Coefficients vs. Time t")
        
        # Time runs from T to 0, so we plot against T-t
        #time_to_maturity = self.terminal_time - self.time_grid
        time_to_maturity = self.time_grid
        
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
            ax.set_xlabel("Time t")
            ax.grid(True)
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def _get_spreads_aux(self, time: float, inventory: float, u: float, ii: int) -> tuple:

        # Get coefficients for the current time
        A, b0, b1, c0, c1, c2 = self.precomputed_A[ii], self.precomputed_b0[ii], self.precomputed_b1[ii], self.precomputed_c0[ii], self.precomputed_c1[ii], self.precomputed_c2[ii]

        print("MBT A:", A, "b0:", b0, "b1:", b1, "c0:", c0, "c1:", c1, "c2:", c2)

        # Calculate B(t,u)
        B = b0 + u * b1
        print("MBT u:", u)
        print("MBT B:", B)
        print("MBT inventory:", inventory)

        h = inventory**2 * A + inventory * (b0 + u * b1) + (c0 + u * c1 + u**2 * c2)
        h_ = (inventory + 1)**2 * A + (inventory + 1) * (b0 + u * b1) + (c0 + u * c1 + u**2 * c2)
        h__ = (inventory - 1)**2 * A + (inventory - 1) * (b0 + u * b1) + (c0 + u * c1 + u**2 * c2)

        print("MBT h:", h, "h_:", h_, "h__:", h__)

        # ask_spread =  1 / self.k - h__ + h
        # bid_spread =  1 / self.k - h_ + h
        
        # return bid_spread, ask_spread
        
        # Use the direct analytical formula for spreads
        ask_spread = (1 / self.k) + (2 * inventory - 1) * A + B
        bid_spread = (1 / self.k) - (2 * inventory + 1) * A - B
        
        return bid_spread, ask_spread


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
        self.n_steps = self.env.n_steps

        self.k = self.env.model_dynamics.fill_probability_model.fill_exponent
        self.sigma = self.env.model_dynamics.midprice_model.volatility
        self.fads_proportion =  self.env.model_dynamics.midprice_model.fads_proportion
        self.p = self.env.model_dynamics.midprice_model.p
        assert isinstance(self.env, TradingEnvironment)
        self.terminal_time = self.env.terminal_time
        self.volatility = self.env.model_dynamics.midprice_model.volatility
        self.time_grid_size = int(self.terminal_time // self.step_size + 1)
        # Create time grid from T to 0
        self.time_grid = np.linspace(0.0, self.terminal_time, self.time_grid_size)

        self._precompute_all_coefficients()
        print(f"✓ Pre-computed ODE solutions for {self.time_grid_size} time points")


    def _precompute_all_coefficients(self):

        self.delta_t = self.terminal_time / (self.n_steps)
        A=np.zeros(self.n_steps)
        b_0 = np.zeros(self.n_steps)
        b_1 = np.zeros(self.n_steps)
        A[-1] = - self.alpha
        c_0 = np.zeros(self.n_steps)
        c_1 = np.zeros(self.n_steps)
        c_2 = np.zeros(self.n_steps)

        self.Gamma = self.gamma * self.sigma * self.fads_proportion

        def x_1(u):  
            return self.mu - self.sigma * self.fads_proportion * self.eta * u

        x_2 = self.sigma  # functions x_2 and x_3
        x_3 = (-1) * self.riccati_solver() * self.fads_proportion * self.eta + self.fads_proportion * np.array(
            [1 for k in range(len(list(self.riccati_solver())))])

        for t in range(self.n_steps - 2, -1, -1):

            A[t] = A[t + 1] + self.delta_t * (
                        - self.big_phi + 4 * (self.psi + self.phi) * np.exp(-1) * self.k * A[t + 1] ** 2)
            b_0[t] = b_0[t + 1] + self.delta_t * (
                        self.mu + 4 * self.k * (self.psi + self.phi) * np.exp(-1) * A[t + 1] * b_0[t + 1])
            b_1[t] = b_1[t + 1] + self.delta_t * (- self.eta * self.sigma * self.fads_proportion - self.eta * b_1[t + 1] + 4 * (
                        self.psi + self.phi) * np.exp(-1)
                      * A[t + 1] * self.k * b_1[t + 1] + 4 * np.exp(-1) * self.psi * self.Gamma * A[t + 1]
                        + 4 * np.exp(-1) * self.k * self.Gamma * self.psi * A[t + 1] ** 2)
            c_2[t] = c_2[t + 1] + self.delta_t * (
                        - 2 * self.eta * c_2[t + 1] + (np.exp(-1) / self.k) * (self.k ** 2 * (self.psi + self.phi
                            ) * b_1[t + 1] ** 2 + 2 * self.k ** 2 * self.psi *
                                self.Gamma * A[t + 1] * b_1[t + 1] + 2 * self.psi * self.k * self.Gamma * b_1[t + 1]))
            c_1[t] = c_1[t + 1] + self.delta_t * ( - self.eta * c_1[t + 1] + np.exp(-1) / self.k *
                                                   (2 * self.psi * self.k * self.Gamma* b_0[t + 1] + self.k ** 2 *
                                                    (self.phi + self.psi) * 2 * b_0[t + 1] * b_1[t + 1] +
                                                    2 * self.k ** 2 * self.psi * self.Gamma * b_0[t + 1] * A[t + 1]))
            c_0[t] = c_0[t + 1] + self.delta_t * (x_3[t+1]**2*c_2[t + 1] + (np.exp(-1) / self.k) * (
                        2 * (self.psi + self.phi) + 2 * self.k * A[t + 1] * (self.phi + self.psi)
                        + self.k ** 2 * (self.phi + self.psi) * (A[t + 1] ** 2 + b_0[t + 1] ** 2)))

        self.precomputed_A = A
        self.precomputed_b0 = b_0
        self.precomputed_b1 = b_1
        self.precomputed_c0 = c_0
        self.precomputed_c1 = c_1
        self.precomputed_c2 = c_2

        # Precompute the full information coefficients as well just for the purpose of comparison

        A_FI=np.zeros(self.n_steps)
        b_0_FI = np.zeros(self.n_steps)
        b_1_FI = np.zeros(self.n_steps)
        A_FI[-1] = - self.alpha
        c_0_FI = np.zeros(self.n_steps)
        c_1_FI = np.zeros(self.n_steps)
        c_2_FI  = np.zeros(self.n_steps)

        for t in range(self.n_steps - 2, -1, -1):
            A_FI[t] = A_FI[t+1] + self.delta_t * (- self.big_phi + 4 * (self.psi+ self.phi) * np.exp(-1) * self.k * A_FI[t+1]**2)
            b_0_FI[t] = b_0_FI[t+1] + self.delta_t * (self.mu + 4 *self.k *(self.psi + self.phi) * np.exp(-1) * A_FI[t+1]*b_0_FI[t+1])
            b_1_FI[t] = b_1_FI[t+1] + self.delta_t * (- self.eta * self.sigma * self.fads_proportion - self.eta * b_1_FI[t+1] + 4*(self.psi + self.phi) * np.exp(-1)
                                                *A_FI[t+1] *self.k*b_1_FI[t+1] +4*np.exp(-1)*self.psi * self.Gamma * A_FI[t+1]
                                                +4*np.exp(-1)*self.k*self.Gamma*self.psi*A_FI[t+1]**2)
            c_2_FI[t] = c_2_FI[t+1] + self.delta_t * (- 2*self.eta*c_2_FI[t+1] + (np.exp(-1)/self.k) * (self.k**2*(self.psi + self.phi
                                                )*b_1_FI[t+1]**2 + 2 * self.k**2 * self.psi *self.Gamma * A_FI[t+1]
                                                        *b_1_FI[t+1] + 2*self.psi * self.k*self.Gamma * b_1_FI[t+1]))
            c_1_FI[t] = c_1_FI[t+1] + self.delta_t * (- self.eta * c_1_FI[t+1] + np.exp(-1)/self.k * (2 * self.psi *self.k* self.Gamma
                                                *b_0_FI[t+1] + self.k**2*(self.phi + self.psi) *2*b_0_FI[t+1]*b_1_FI[t+1]
                                                    +2*self.k**2 * self.psi*self.Gamma*b_0_FI[t+1]*A_FI[t+1]))
            c_0_FI[t] = c_0_FI[t+1] + self.delta_t *(c_2_FI[t+1] + (np.exp(-1)/self.k) *( 2*(self.psi + self.phi) + 2*self.k*A_FI[t+1]*(self.phi+self.psi)
                                             + self.k**2 *(self.phi+self.psi)*(A_FI[t+1]**2 + b_0_FI[t+1]**2)))

        self.precomputed_A_FI = A_FI
        self.precomputed_b0_FI = b_0_FI
        self.precomputed_b1_FI = b_1_FI
        self.precomputed_c0_FI = c_0_FI
        self.precomputed_c1_FI = c_1_FI
        self.precomputed_c2_FI = c_2_FI

    #################################################################################
    # Helper function for obtaining coefficients

    def riccati_eq(self, t, P):
        term1 = -2 * (self.eta * P * (1 + self.fads_proportion * (P * self.eta * self.fads_proportion - self.fads_proportion)))
        term2 = (1 + (P * self.eta * self.fads_proportion - self.fads_proportion) * self.fads_proportion) ** 2
        term3 = (P * self.eta * self.fads_proportion - self.fads_proportion) ** 2 * self.p ** 2
        return term1 + term2 + term3

    def rk4_step(self, f, t, P, dt):
        k1 = f(t, P)
        k2 = f(t + 0.5 * dt, P + 0.5 * dt * k1)
        k3 = f(t + 0.5 * dt, P + 0.5 * dt * k2)
        k4 = f(t + dt, P + dt * k3)
        return P + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def riccati_solver(self): #computation of the Riccato equation
        P0 = 0.0
        t0 = 0.0
        dt = self.delta_t
        P_values = np.zeros(self.n_steps)
        for i in range(self.n_steps-1):
            P_values[i + 1] = self.rk4_step(self.riccati_eq, self.time_grid[i], P_values[i], dt)
        return P_values
    #################################################################################
  
    def get_action(self, state: np.ndarray):
        action = self._get_action(state)
        if action.min() < 0:
            warnings.warn("MM agent is quoting a negative spread")
        return action

    def _get_spreads(self, state: np.ndarray) -> float:
        value_function_q,value_function_q_pos, value_function_q_neg = self._approximate_value_functions(state, inventories_add=[-1, 0, 1])
        
        ask_spread = 1/self.k - value_function_q_neg + value_function_q
        bid_spread = 1/self.k - value_function_q_pos + value_function_q
        return [bid_spread, ask_spread]
    
    def _get_action(self, state: np.ndarray):
        bid_half_spread, ask_half_spread = self._get_spreads(state)
        bid_half_spread = bid_half_spread.reshape(-1, 1)
        ask_half_spread = ask_half_spread.reshape(-1, 1)
        return np.concatenate([bid_half_spread, ask_half_spread], axis=1)

    def _approximate_value_functions(self, state: np.ndarray, inventories_add=[-1, 0, 1]):

        inventory = state[:, INVENTORY_INDEX]
        time = state[:, TIME_INDEX]
        market_state = state[:, FILTERED_FADS_INDEX]
        
        current_time = time[0]  # Use first element since all are the same
        
        # # Convert current_time to the corresponding index in the precomputed arrays
        # Since time_grid goes from 0 to terminal_time, we can calculate the index directly
        time_index = int(round(current_time / self.step_size))
        time_index = min(time_index, len(self.precomputed_A) - 1)  # Ensure index doesn't exceed array bounds
        
        # Get coefficients directly from precomputed arrays using the time index
        A = self.precomputed_A[time_index]
        b0 = self.precomputed_b0[time_index]
        b1 = self.precomputed_b1[time_index]
        c0 = self.precomputed_c0[time_index]
        c1 = self.precomputed_c1[time_index]
        c2 = self.precomputed_c2[time_index]
              
        value_function = inventory ** 2 * A + inventory * (b0 + market_state * b1) + (c0 + market_state * c1 + market_state ** 2 * c2)
        value_function_pos = (inventory + 1) ** 2 * A + (inventory + 1) * (b0 + market_state * b1) + (c0 + market_state * c1 + market_state ** 2 * c2)
        value_function_neg = (inventory - 1) ** 2 * A + (inventory - 1) * (b0 + market_state * b1) + (c0 + market_state * c1 + market_state ** 2 * c2)
        return value_function, value_function_pos, value_function_neg
    
    #################################################################################
    #  Visualization methods for debugging and analysis

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

    def plot_precomputed_coefficient_differences(self):
        """
        Visualizes the differences between PI and FI precomputed ODE coefficients.
        Shows the difference ('PI' - 'FI') between the solutions.
        """
        fig, axs = plt.subplots(3, 2, figsize=(12, 12), sharex=True)
        fig.suptitle("Difference between PI and FI ODE Coefficients vs Time-to-Maturity")

        # Calculate differences (PI - FI)
        diff_A = self.precomputed_A - self.precomputed_A_FI
        diff_b0 = self.precomputed_b0 - self.precomputed_b0_FI
        diff_b1 = self.precomputed_b1 - self.precomputed_b1_FI
        diff_c0 = self.precomputed_c0 - self.precomputed_c0_FI
        diff_c1 = self.precomputed_c1 - self.precomputed_c1_FI
        diff_c2 = self.precomputed_c2 - self.precomputed_c2_FI

        # Time runs from T to 0, so we plot against T-t
        time_to_maturity = self.terminal_time - self.time_grid

        # Row 1
        axs[0, 0].plot(time_to_maturity, diff_A, 'r-', label="A_PI(t) - A_FI(t)")
        axs[0, 0].set_title("Difference in A(t)")
        axs[0, 0].legend()

        axs[0, 1].plot(time_to_maturity, diff_b0, 'g-', label="b0_PI(t) - b0_FI(t)")
        axs[0, 1].set_title("Difference in b0(t)")
        axs[0, 1].legend()

        # Row 2
        axs[1, 0].plot(time_to_maturity, diff_b1, 'b-', label="b1_PI(t) - b1_FI(t)")
        axs[1, 0].set_title("Difference in b1(t)")
        axs[1, 0].legend()

        axs[1, 1].plot(time_to_maturity, diff_c0, 'm-', label="c0_PI(t) - c0_FI(t)")
        axs[1, 1].set_title("Difference in c0(t)")
        axs[1, 1].legend()

        # Row 3
        axs[2, 0].plot(time_to_maturity, diff_c1, 'c-', label="c1_PI(t) - c1_FI(t)")
        axs[2, 0].set_title("Difference in c1(t)")
        axs[2, 0].legend()

        axs[2, 1].plot(time_to_maturity, diff_c2, 'y-', label="c2_PI(t) - c2_FI(t)")
        axs[2, 1].set_title("Difference in c2(t)")
        axs[2, 1].legend()

        # Add a horizontal line at y=0 for reference
        for ax in axs.flat:
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax.set_xlabel("Time to Maturity (T-t)")
            ax.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def _get_spreads_aux(self, time: float, inventory: float, u: float, ii: int) -> tuple:

        value_function_q, value_function_q_pos, value_function_q_neg = self._approximate_value_functions_(time, inventory, u, ii=ii)
        print("MBT value_function_q:", value_function_q, "value_function_q_pos:", value_function_q_pos, "value_function_q_neg:", value_function_q_neg)
        ask_spread = 1/self.k - value_function_q_neg + value_function_q
        bid_spread = 1/self.k - value_function_q_pos + value_function_q
        return [bid_spread, ask_spread]
    
    def _approximate_value_functions_(self, time: float, inventory: float, u: float, ii: int=0  ):

        print("Current time:", time)
        A, b0, b1, c0, c1, c2 = self.precomputed_A[ii], self.precomputed_b0[ii], self.precomputed_b1[ii], self.precomputed_c0[ii], self.precomputed_c1[ii], self.precomputed_c2[ii]
        
        print("MBT A, b0, b1, c0, c1, c2:", A, b0, b1, c0, c1, c2)
              
        value_function = inventory ** 2 * A + inventory * (b0 + u * b1) + (c0 + u * c1 + u ** 2 * c2)
        value_function_pos = (inventory + 1) ** 2 * A + (inventory + 1) * (b0 + u * b1) + (c0 + u * c1 + u ** 2 * c2)
        value_function_neg = (inventory - 1) ** 2 * A + (inventory - 1) * (b0 + u * b1) + (c0 + u * c1 + u ** 2 * c2)
        return value_function, value_function_pos, value_function_neg
 