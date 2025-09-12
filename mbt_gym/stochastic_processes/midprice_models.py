from math import sqrt
from typing import Optional

import numpy as np

from mbt_gym.stochastic_processes.StochasticProcessModel import StochasticProcessModel

MidpriceModel = StochasticProcessModel

from mbt_gym.gym.index_names import BID_INDEX, ASK_INDEX, TIME_INDEX

class ConstantMidpriceModel(MidpriceModel):
    def __init__(
        self,
        initial_price: float = 100,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.terminal_time = terminal_time
        super().__init__(
            min_value=np.array([[initial_price]]),
            max_value=np.array([[initial_price]]),
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=np.array([[initial_price]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None) -> np.ndarray:
        pass


class BrownianMotionMidpriceModel(MidpriceModel):
    def __init__(
        self,
        drift: float = 0.0,
        volatility: float = 2.0,
        initial_price: float = 100,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.drift = drift
        self.volatility = volatility
        self.terminal_time = terminal_time
        super().__init__(
            min_value=np.array([[initial_price - (self._get_max_value(initial_price, terminal_time) - initial_price)]]),
            max_value=np.array([[self._get_max_value(initial_price, terminal_time)]]),
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=np.array([[initial_price]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None) -> np.ndarray:
        self.current_state = (
            self.current_state
            + self.drift * self.step_size * np.ones((self.num_trajectories, 1))
            + self.volatility * sqrt(self.step_size) * self.rng.normal(size=(self.num_trajectories, 1))
        )

    def _get_max_value(self, initial_price, terminal_time):
        return initial_price + 4 * self.volatility * np.sqrt(terminal_time)


class GeometricBrownianMotionMidpriceModel(MidpriceModel):
    def __init__(
        self,
        drift: float = 0.0,
        volatility: float = 0.1,
        initial_price: float = 100,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.drift = drift
        self.volatility = volatility
        self.terminal_time = terminal_time
        super().__init__(
            min_value=np.array([[initial_price - (self._get_max_value(initial_price, terminal_time) - initial_price)]]),
            max_value=np.array([[self._get_max_value(initial_price, terminal_time)]]),
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=np.array([[initial_price]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None) -> np.ndarray:
        self.current_state = (
            self.current_state
            + self.drift * self.current_state * self.step_size
            + self.volatility
            * self.current_state
            * sqrt(self.step_size)
            * self.rng.normal(size=(self.num_trajectories, 1))
        )

    def _get_max_value(self, initial_price, terminal_time):
        stdev = sqrt(
            initial_price**2
            * np.exp(2 * self.drift * terminal_time)
            * (np.exp(self.volatility**2 * terminal_time) - 1)
        )
        return initial_price * np.exp(self.drift * terminal_time) + 4 * stdev


class OuMidpriceModel(MidpriceModel):
    def __init__(
        self,
        mean_reversion_level: float = 0.0,
        mean_reversion_speed: float = 1.0,
        volatility: float = 2.0,
        initial_price: float = 100.0,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.mean_reversion_level = mean_reversion_level
        self.mean_reversion_speed = mean_reversion_speed
        self.volatility = volatility
        self.terminal_time = terminal_time
        super().__init__(
            min_value=np.array([[initial_price - (self._get_max_value(initial_price, terminal_time) - initial_price)]]),
            max_value=np.array([[self._get_max_value(initial_price, terminal_time)]]),
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=np.array([[initial_price]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None) -> np.ndarray:
        self.current_state += -self.mean_reversion_speed * (
            self.current_state - self.mean_reversion_level * np.ones((self.num_trajectories, 1))
        ) + self.volatility * sqrt(self.step_size) * self.rng.normal(size=(self.num_trajectories, 1))

    def _get_max_value(self, initial_price, terminal_time):
        return initial_price + 4 * self.volatility * terminal_time  # TODO: What should this be?


class ShortTermOuAlphaMidpriceModel(MidpriceModel):
    def __init__(
        self,
        volatility: float = 2.0,
        ou_process: OuMidpriceModel = None,
        initial_price: float = 100.0,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.volatility = volatility
        self.ou_process = ou_process or OuMidpriceModel(initial_price=0.0)
        self.terminal_time = terminal_time
        super().__init__(
            min_value=np.array(
                [
                    [
                        initial_price - (self._get_max_asset_price(initial_price, terminal_time) - initial_price),
                        self.ou_process.min_value,
                    ]
                ]
            ),
            max_value=np.array([[self._get_max_asset_price(initial_price, terminal_time), self.ou_process.max_value]]),
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=np.array([[initial_price, self.ou_process.initial_state[0][0]]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None) -> np.ndarray:
        self.current_state[:, 0] = (
            self.current_state[:, 0]
            + self.ou_process.current_state * self.step_size * np.ones((self.num_trajectories, 1))
            + self.volatility * sqrt(self.step_size) * self.rng.normal(size=(self.num_trajectories, 1))
        )
        self.ou_process.update(arrivals, fills, actions)
        self.current_state[:, 1] = self.ou_process.current_state

    def _get_max_asset_price(self, initial_price, terminal_time):
        return initial_price + 4 * self.volatility * terminal_time  # TODO: what should this be?


class BrownianMotionJumpMidpriceModel(MidpriceModel):
    def __init__(
        self,
        drift: float = 0.0,
        volatility: float = 2.0,
        jump_size: float = 1.0,
        initial_price: float = 100,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.drift = drift
        self.volatility = volatility
        self.jump_size = jump_size
        self.terminal_time = terminal_time
        super().__init__(
            min_value=np.array([[initial_price - (self._get_max_value(initial_price, terminal_time) - initial_price)]]),
            max_value=np.array([[self._get_max_value(initial_price, terminal_time)]]),
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=np.array([[initial_price]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None) -> np.ndarray:
        fills_bid = fills[:, BID_INDEX] * arrivals[:, BID_INDEX]
        fills_ask = fills[:, ASK_INDEX] * arrivals[:, ASK_INDEX]
        self.current_state = (
            self.current_state
            + self.drift * self.step_size * np.ones((self.num_trajectories, 1))
            + self.volatility * sqrt(self.step_size) * self.rng.normal(size=(self.num_trajectories, 1))
            + (self.jump_size * fills_ask - self.jump_size * fills_bid).reshape(-1,1)
        )

    def _get_max_value(self, initial_price, terminal_time):
        return initial_price + 4 * self.volatility * terminal_time


class OuJumpMidpriceModel(MidpriceModel):
    def __init__(
        self,
        mean_reversion_level: float = 0.0,
        mean_reversion_speed: float = 1.0,
        volatility: float = 2.0,
        jump_size: float = 1.0,
        initial_price: float = 100.0,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.mean_reversion_level = mean_reversion_level
        self.mean_reversion_speed = mean_reversion_speed
        self.volatility = volatility
        self.jump_size = jump_size
        self.terminal_time = terminal_time
        super().__init__(
            min_value=np.array([[initial_price - (self._get_max_value(initial_price, terminal_time) - initial_price)]]),
            max_value=np.array([[self._get_max_value(initial_price, terminal_time)]]),
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=np.array([[initial_price]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None) -> np.ndarray:
        fills_bid = fills[:, BID_INDEX] * arrivals[:, BID_INDEX]
        fills_ask = fills[:, ASK_INDEX] * arrivals[:, ASK_INDEX]
        self.current_state = (
            self.current_state
            - self.mean_reversion_speed
            * (self.current_state - self.mean_reversion_level * np.ones((self.num_trajectories, 1)))
            + self.volatility * sqrt(self.step_size) * self.rng.normal(size=(self.num_trajectories, 1))            
            + (self.jump_size * fills_ask - self.jump_size * fills_bid).reshape(-1,1)
        )

    def _get_max_value(self, initial_price, terminal_time):
        return initial_price + 4 * self.volatility * terminal_time


class ShortTermJumpAlphaMidpriceModel(MidpriceModel):
    def __init__(
        self,
        volatility: float = 2.0,
        ou_jump_process: OuJumpMidpriceModel = None,
        initial_price: float = 100.0,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.volatility = volatility
        self.ou_jump_process = ou_jump_process or OuJumpMidpriceModel(initial_price=0.0)
        self.terminal_time = terminal_time
        super().__init__(
            min_value=np.array(
                [
                    [
                        initial_price - (self._get_max_asset_price(initial_price, terminal_time) - initial_price),
                        self.ou_jump_process.min_value,
                    ]
                ]
            ),
            max_value=np.array(
                [[self._get_max_asset_price(initial_price, terminal_time), self.ou_jump_process.max_value]]
            ),
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=np.array([[initial_price, self.ou_jump_process.initial_state[0][0]]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None) -> np.ndarray:
        self.current_state[:, 0] = (
            self.current_state[:, 0]
            + self.ou_jump_process.current_state * self.step_size * np.ones((self.num_trajectories, 1))
            + self.volatility * sqrt(self.step_size) * self.rng.normal(size=(self.num_trajectories, 1))
        )
        self.ou_jump_process.update(arrivals, fills, actions)
        self.current_state[:, 1] = self.ou_jump_process.current_state

    def _get_max_asset_price(self, initial_price, terminal_time):
        return initial_price + 4 * self.volatility * terminal_time  # TODO: what should this be?


class HestonMidpriceModel(MidpriceModel):
    # Current/Initial State with the Heston model will consist of price AND current variance, not just price
    def __init__(
        self,
        drift: float = 0.05,
        volatility_mean_reversion_rate: float = 3,
        volatility_mean_reversion_level: float = 0.04,
        weiner_correlation: float = -0.8,
        volatility_of_volatility: float = 0.6,
        initial_price: float = 100,
        initial_variance: float = 0.2**2,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.drift = drift
        self.volatility_mean_reversion_rate = volatility_mean_reversion_rate
        self.terminal_time = terminal_time
        self.weiner_correlation = weiner_correlation
        self.volatility_mean_reversion_level = volatility_mean_reversion_level
        self.volatility_of_volatility = volatility_of_volatility
        super().__init__(
            min_value=np.array([[initial_price - (self._get_max_value(initial_price, terminal_time) - initial_price)]]),
            max_value=np.array([[self._get_max_value(initial_price, terminal_time)]]),
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=np.array([[initial_price, initial_variance]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None) -> np.ndarray:
        weiner_means = np.array([0, 0])
        weiner_corr = np.array([[1, self.weiner_correlation], [self.weiner_correlation, 1]])
        weiners = np.random.multivariate_normal(weiner_means, cov=weiner_corr, size=self.num_trajectories)
        self.current_state[:, 0] = (
            self.current_state[:, 0]
            + self.drift * self.current_state[:, 0] * self.step_size
            + np.sqrt(self.current_state[:, 1] * self.step_size) * self.current_state[:, 0] * weiners[:, 0]
        )
        self.current_state[:, 1] = np.abs(
            self.current_state[:, 1]
            + self.volatility_mean_reversion_rate
            * (self.volatility_mean_reversion_level - self.current_state[:, 1])
            * self.step_size
            + self.volatility_of_volatility * np.sqrt(self.current_state[:, 1] * self.step_size) * weiners[:, 1]
        )

    def _get_max_value(self, initial_price, terminal_time):
        return initial_price + 4 * self.volatility_mean_reversion_level * terminal_time


class ConstantElasticityOfVarianceMidpriceModel(MidpriceModel):
    def __init__(
        self,
        drift: float = 0.0,
        volatility: float = 0.1,
        gamma: float = 1,  # gamma = 1 is just gbm
        initial_price: float = 100,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.drift = drift
        self.volatility = volatility
        self.gamma = gamma
        self.terminal_time = terminal_time
        super().__init__(
            min_value=np.array([[initial_price - (self._get_max_value(initial_price, terminal_time) - initial_price)]]),
            max_value=np.array([[self._get_max_value(initial_price, terminal_time)]]),
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=np.array([[initial_price]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None) -> np.ndarray:
        self.current_state = (
            self.current_state
            + self.current_state * self.drift * self.step_size  # *np.ones((self.num_trajectories, 1))
            + self.volatility
            * (self.current_state**self.gamma)
            * np.sqrt(self.step_size)
            * np.random.normal(size=self.num_trajectories)
        )

    def _get_max_value(self, initial_price, terminal_time):
        return initial_price + 4 * self.volatility * terminal_time


class ArithmeticBrownianMotionWithFadsMidpriceModel(MidpriceModel):
    def __init__(
        self,
        drift: float = 0.0,
        volatility: float = 1.0,
        fads_proportion: float = 1,  # fad relevance parameter (which directly define p)
        eta: float = 10,  # fad mean reversion speed
        initial_price: float = 100,
        initial_fad: float = 0,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.drift = drift
        self.volatility = volatility
        self.terminal_time = terminal_time
        self.fads_proportion = fads_proportion
        self.eta = eta
        
        # directly identified paramters
        self.p = np.sqrt(1-self.fads_proportion**2)
        # self.psi = (30 - phi * self.terminal_time ) / ... # ???
        super().__init__(
            min_value=np.array([[initial_price - (self._get_max_value(initial_price, terminal_time) - initial_price)]]),
            max_value=np.array([[self._get_max_value(initial_price, terminal_time)]]),
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=np.array([[initial_price, initial_fad]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None) -> np.ndarray:
        # Sample both independent increments simultaneously
        random_increments =  self.rng.normal(size=(self.num_trajectories, 2))
        dB_fad = random_increments[:, 0:1]  # First column for fad
        dZ_martingale = random_increments[:, 1:2] * np.sqrt(self.step_size) # Second column for Z martingale

        # " Alternative implementation for the OU component:"
        # --- Euler-Maruyama discretization of the OU process ---
        #self.current_state[:, 1] = self.current_state[:, 1] - self.eta * self.current_state[:, 1] * self.step_size + dB_fad.flatten()

        # --- Exact OU update for fad ---
        exp_term = np.exp(-self.eta * self.step_size)
        std_term = np.sqrt((1 - np.exp(-2 * self.eta * self.step_size)) / (2 * self.eta))
        # TODO: attention here
        #self.current_state[:, 1] = (self.current_state[:, 1] * exp_term + std_term * np.sqrt(self.step_size) * dB_fad.flatten())
        self.current_state[:, 1] = (self.current_state[:, 1] * exp_term + std_term * dB_fad.flatten())

        # Stochastic part of the midprice update
        self.stochastic_part = (self.p * dZ_martingale).flatten() + self.fads_proportion * self.current_state[:, 1]

        self.current_state[:, 0] = self.current_state[:, 0] + (
            self.drift * self.step_size
            + self.volatility * self.stochastic_part
        )

    def _get_max_value(self, initial_price, terminal_time):
        # we are assuming that max value is given related just to the brownian motion part without considering the fads
        return initial_price + 4 * self.volatility * terminal_time



class ArithmeticBrownianMotionWithFadsMidpriceModelPartialInformation(MidpriceModel):
    def __init__(
        self,
        drift: float = 0.0,
        volatility: float = 1.0,
        fads_proportion: float = 1,  # fad relevance parameter (which directly define p)
        eta: float = 10,  # fad mean reversion speed
        initial_price: float = 100,
        initial_fad: float = 0,
        initial_filtered_fad: float = 0,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.drift = drift
        self.volatility = volatility
        self.terminal_time = terminal_time
        self.fads_proportion = fads_proportion
        self.eta = eta
        
        # directly identified paramters
        self.p = np.sqrt(1-self.fads_proportion**2)

        self.P_hat_array = self.compute_P_hat_path(eta=self.eta, q=self.fads_proportion, p=self.p, T=self.terminal_time, dt=step_size)
        # self.psi = (30 - phi * self.terminal_time ) / ... # ???
        super().__init__(
            min_value=np.array([[initial_price - (self._get_max_value(initial_price, terminal_time) - initial_price)]]),
            max_value=np.array([[self._get_max_value(initial_price, terminal_time)]]),
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=np.array([[initial_price, initial_fad, initial_filtered_fad]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None) -> np.ndarray:
        time = state[:, TIME_INDEX]      
        current_time = time[0]  # Use first element since all are the same
        
        # Sample both independent increments simultaneously
        random_increments =  self.rng.normal(size=(self.num_trajectories, 3))
        dB_fad = random_increments[:, 0:1]  # First column for fad
        dZ_martingale = random_increments[:, 1:2] * np.sqrt(self.step_size) # Second column for Z martingale

        # " Alternative implementation for the OU component:"
        # --- Euler-Maruyama discretization of the OU process ---
        #self.current_state[:, 1] = self.current_state[:, 1] - self.eta * self.current_state[:, 1] * self.step_size + dB_fad.flatten()

        # --- Exact OU update for fad ---
        exp_term = np.exp(-self.eta * self.step_size)
        std_term = np.sqrt((1 - np.exp(-2 * self.eta * self.step_size)) / (2 * self.eta))

        # self.current_state[:, 1] = (self.current_state[:, 1] * exp_term + std_term * np.sqrt(self.step_size) * dB_fad.flatten())
        self.current_state[:, 1] = (self.current_state[:, 1] * exp_term + std_term * dB_fad.flatten())

        # Stochastic part of the midprice update
        self.stochastic_part = (self.p * dZ_martingale).flatten() + self.fads_proportion * self.current_state[:, 1]
        
        dS = self.drift * self.step_size + self.volatility * self.stochastic_part
        
        self.current_state[:, 0] = self.current_state[:, 0] + dS
        
        # In the partial information setting, we do not observe complete the fad component, but just filtered version

        # Update P_hat via Riccati ODE (Euler step)
        # dP_dt = - (self.eta**2 * self.fads_proportion**2) * (self.P_hat**2) \
        #         - self.P_hat * (2*self.eta - 2*self.eta*self.fads_proportion**2) \
        #         + self.p**2
        # self.P_hat += dP_dt * self.step_size
        self.P_hat = self.P_hat_array[int(current_time / self.step_size)]

        # given: dS (observed or simulated), dt, and current U_hat, P_hat
        pi_h = self.drift - self.eta * self.fads_proportion * self.volatility * self.current_state[:, 2]      # π_t(h)
        dI = dS - pi_h * self.step_size                       # innovation increment

        # Update U_hat via SDE (Euler-Maruyama step)
        dU_hat = -self.eta * self.current_state[:, 2] * self.step_size \
             + (1/self.volatility) * (-self.eta*self.fads_proportion*self.P_hat + self.fads_proportion) * dI
        self.current_state[:, 2] = self.current_state[:, 2] + dU_hat
        # Debug prints
        # print(f"time: {current_time:.5f}")
        # print(f"U_hat before: {self.current_state[:,2]}")
        # print(f"P_hat: {self.P_hat}")
        # print(f"dS: {dS}")
        # print(f"dI (innovation): {dI}")
        # K = (-self.eta*self.fads_proportion*self.P_hat + self.fads_proportion)/self.volatility
        # print(f"Kalman gain factor: {K}")
        # print(f"Euler term: {-self.eta*self.current_state[:,2]*self.step_size}")
        # print(f"dU_hat (update increment): {dU_hat}")

        self.U_hat_history = []
        self.fad_history = []

        self.U_hat_history.append(self.current_state[:,2].copy())
        self.fad_history.append(self.current_state[:,1].copy())

    def _get_max_value(self, initial_price, terminal_time):
        # we are assuming that max value is given related just to the brownian motion part without considering the fads
        return initial_price + 4 * self.volatility * terminal_time


    def compute_P_hat_path(self, eta: float, q: float, p: float, T: float, dt: float) -> np.ndarray:
        """
        Compute the conditional variance path P_hat(t) for t in [0, T]
        using closed-form Riccati solution when stable,
        and falling back to Euler discretization otherwise.

        Parameters
        ----------
        eta : float
            Mean reversion speed.
        q : float
            Fads proportion.
        p : float
            Complementary factor (p^2 + q^2 = 1).
        T : float
            Terminal time.
        dt : float
            Step size.

        Returns
        -------
        P_path : np.ndarray
            Array of size (num_steps+1,) with values of P_hat(t).
        """
        a = (eta**2) * (q**2)
        b = 2*eta*(1 - q**2)
        c = p**2

        num_steps = int(np.ceil(T / dt))
        t_grid = np.linspace(0, num_steps*dt, num_steps+1)

        Delta = b*b + 4*a*c
        sqrtD = np.sqrt(max(Delta, 0.0))  # safeguard

        # If a ≈ 0, reduce to linear ODE
        if abs(a) < 1e-14:
            P_path = np.zeros_like(t_grid)
            if abs(b) < 1e-14:
                # P(t) = c t
                P_path = c * t_grid
            else:
                # P(t) = (c/b)(1 - exp(-b t))
                P_path = (c / b) * (1 - np.exp(-b * t_grid))
            return P_path

        # Try closed form
        r1 = (-b + sqrtD) / (2*a)
        r2 = (-b - sqrtD) / (2*a)
        lam = sqrtD  # since lam = a(r1 - r2) = sqrt(D)

        if lam > 1e-12 and abs(r2) > 1e-12:
            print("Using closed-form solution for P_hat(t)")
            exp_term = np.exp(-lam * t_grid)
            numerator = r1 * r2 * (1.0 - exp_term)
            denominator = r2 - r1 * exp_term
            P_path = numerator / denominator
            return np.maximum(P_path, 0.0)  # clip negatives from numerical noise

        # Fallback: Euler discretization
        P_path = np.zeros_like(t_grid)
        P = 0.0
        print("Using Euler discretization for P_hat(t)")
        for i in range(num_steps):
            dPdt = - a * P**2 - b * P + c
            P += dPdt * dt
            P_path[i+1] = P
        return np.maximum(P_path, 0.0)

