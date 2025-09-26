import abc
from typing import Optional

import numpy as np

from mbt_gym.gym.index_names import ASSET_PRICE_INDEX, FADS_INDEX
from mbt_gym.stochastic_processes.StochasticProcessModel import StochasticProcessModel


class ArrivalModel(StochasticProcessModel):
    """ArrivalModel models the arrival of orders to the order book. The first entry of arrivals represents an arrival
    of an exogenous SELL order (arriving on the buy side of the book) and the second entry represents an arrival of an
    exogenous BUY order (arriving on the sell side of the book).
    """

    def __init__(
        self,
        min_value: np.ndarray,
        max_value: np.ndarray,
        step_size: float,
        terminal_time: float,
        initial_state: np.ndarray,
        num_trajectories: int = 1,
        seed: int = None,
    ):
        super().__init__(min_value, max_value, step_size, terminal_time, initial_state, num_trajectories, seed)

    @abc.abstractmethod
    def get_arrivals(self) -> np.ndarray:
        pass


class PoissonArrivalModel(ArrivalModel):
    def __init__(
        self,
        intensity: np.ndarray = np.array([140.0, 140.0]),
        step_size: float = 0.001,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.intensity = np.array(intensity)
        super().__init__(
            min_value=np.array([[]]),
            max_value=np.array([[]]),
            step_size=step_size,
            terminal_time=0.0,
            initial_state=np.array([[]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None):
        pass

    def get_arrivals(self) -> np.ndarray:
        unif = self.rng.uniform(size=(self.num_trajectories, 2))    # geneerate 2 numbers for trajectories (1 for buy 1 for sell)
        return unif < self.intensity * self.step_size


class PoissonArrivalNonLinearModel(ArrivalModel):
    def __init__(
        self,
        intensity: np.ndarray = np.array([140.0, 140.0]),
        step_size: float = 0.001,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.intensity = np.array(intensity)
        super().__init__(
            min_value=np.array([[]]),
            max_value=np.array([[]]),
            step_size=step_size,
            terminal_time=0.0,
            initial_state=np.array([[]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None):
        pass

    def get_arrivals(self) -> np.ndarray:
        unif = self.rng.uniform(size=(self.num_trajectories, 2))
        return unif < 1. - np.exp(-self.intensity * self.step_size)


class HawkesArrivalModel(ArrivalModel):
    def __init__(
        self,
        baseline_arrival_rate: np.ndarray = np.array([[10.0, 10.0]]),
        step_size: float = 0.01,
        jump_size: float = 40.0,
        mean_reversion_speed: float = 60.0,
        terminal_time: float = 1,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.baseline_arrival_rate = baseline_arrival_rate
        self.jump_size = jump_size  # see https://arxiv.org/pdf/1507.02822.pdf, equation (4).
        self.mean_reversion_speed = mean_reversion_speed
        super().__init__(
            min_value=np.array([[0, 0]]),
            max_value=np.array([[1, 1]]) * self._get_max_arrival_rate(),
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=baseline_arrival_rate,
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None) -> np.ndarray:
        self.current_state = (
            self.current_state
            + self.mean_reversion_speed
            * (np.ones((self.num_trajectories, 2)) * self.baseline_arrival_rate - self.current_state)
            * self.step_size
            * np.ones((self.num_trajectories, 2))
            + self.jump_size * arrivals
        )
        return self.current_state

    def get_arrivals(self) -> np.ndarray:
        unif = self.rng.uniform(size=(self.num_trajectories, 2))
        return unif < self.current_state * self.step_size

    def _get_max_arrival_rate(self):
        return self.baseline_arrival_rate * 10

    # TODO: Improve this with 4*std
    # See: https://math.stackexchange.com/questions/4047342/expectation-of-hawkes-process-with-exponential-kernel

class FadsInformedUniformedTradersArrivalModel(ArrivalModel):
    def __init__(
        self,
        step_size: float = 0.01,
        phi: float = 15,
        psi: float = 15,
        k: float = 0.5,
        gamma: float = 0.5,
        fads_proportion: float = 0.5,
        sigma: float = 0.5,
        terminal_time: float = 1,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.baseline_arrival_rate = np.array([[phi + psi, phi + psi]])
        self.phi = phi
        self.psi = psi
        self.k = k
        self.gamma = gamma
        self.fads_proportion = fads_proportion
        self.sigma = sigma
        super().__init__(
            min_value=np.array([[0, 0]]),
            max_value=np.array([[1, 1]]) * self._get_max_arrival_rate(),
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=self.baseline_arrival_rate,
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None) -> np.ndarray:
        
        # current state is (num_trajectories, 2) meaning for each trajectory we have a buy and sell arrival rate
        # at every step (so time istante) we update the arrival rate according to the formula, so will depend on spread of previous round and the fad
        S_minus=-np.inf
        S_plus=+np.inf

        self.current_state[:,0] = self.phi * np.exp(-self.k * actions[:, 0]) + self.psi * np.exp(-self.k * actions[:, 0] - self.gamma * (self.sigma * self.fads_proportion * np.maximum(state[:, FADS_INDEX], S_minus)))
        self.current_state[:,1] = self.phi * np.exp(-self.k * actions[:, 1]) + self.psi * np.exp(-self.k * actions[:, 1] + self.gamma * (self.sigma * self.fads_proportion * np.minimum(state[:, FADS_INDEX], S_plus)))
        if np.any(self.current_state < 0):
            print("Warning: Negative arrival rate encountered. Setting to zero.")
        return self.current_state

    def get_arrivals(self) -> np.ndarray:
        unif = self.rng.uniform(size=(self.num_trajectories, 2))
        return unif < self.current_state * self.step_size

    def _get_max_arrival_rate(self):
        return 100 # self.baseline_arrival_rate * 10 TODO: improve this 


class ModifiedPoissonArrivalModel(ArrivalModel):
    def __init__(
        self,
        phi: float = 15,
        psi: float = 15,
        gamma: float = 0.5,
        fads_proportion: float = 0.5,
        sigma: float = 0.5,
        step_size: float = 0.001,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):  
        self.phi = phi
        self.psi = psi
        self.gamma = gamma
        self.fads_proportion = fads_proportion
        self.sigma = sigma

         # Initialize initial_state with correct shape (num_trajectories, 2)
        #initial_state = np.zeros((1, 2))
        initial_state = np.ones((1, 2)) * (self.phi + self.psi) 


        super().__init__(
            min_value=np.array([[]]),
            max_value=np.array([[]]),
            step_size=step_size,
            terminal_time=0.0,
            initial_state=initial_state,
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None):
        S_minus = -np.inf
        S_plus = +np.inf
        informed_intensity_ask = self.psi * np.exp(-self.gamma *  np.maximum(self.fads_proportion * self.sigma * state[:, FADS_INDEX], S_minus))
        informed_intensity_bid = self.psi * np.exp(+self.gamma *  np.minimum(self.fads_proportion * self.sigma * state[:, FADS_INDEX], S_plus))
        # Stack as columns to get shape (num_trajectories, 2)
        self.current_state = np.column_stack([self.phi + informed_intensity_bid, self.phi + informed_intensity_ask])
        return self.current_state
    
    def get_arrivals(self) -> np.ndarray:
        unif = self.rng.uniform(size=(self.num_trajectories, 2)) 
        # print("current_state shape:", self.current_state.shape)
        # print("current_state mean:", np.mean(self.current_state, axis=0))
        return unif < self.current_state * self.step_size
