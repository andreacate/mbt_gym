from math import sqrt
from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp

from mbt_gym.stochastic_processes.StochasticProcessModel import StochasticProcessModel

MidpriceModel = StochasticProcessModel

from mbt_gym.gym.index_names import BID_INDEX, ASK_INDEX, TIME_INDEX, INVENTORY_INDEX

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
        return initial_price + 4 * self.volatility * terminal_time  

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
        return initial_price + 4 * self.volatility * terminal_time 


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
        return initial_price + 4 * self.volatility * terminal_time 


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
        fads_proportion: float = 0.8,  
        eta: float = 10, 
        initial_price: float = 100,
        initial_fad: float = 0,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
        use_filtering: bool = True,  # Toggle between complete/partial information
    ):
        self.drift = drift  
        self.volatility = volatility  
        self.fads_proportion = fads_proportion 
        self.eta = eta  
        self.terminal_time = terminal_time
        self.use_filtering = use_filtering
        
        self.p = np.sqrt(1 - self.fads_proportion**2)  
   
        # Initialize filtering variables
        self.filtered_fad = None  # Û_t = E[U_t | F^S_t]
        self.conditional_variance = None  # P̂_t = E[(U_t - Û_t)²]
        self.innovation_process = None  # I_t
        
        # State vector: [S_t, U_t, Û_t]
        initial_filtered_fad = initial_fad 
        
        super().__init__(
            min_value=np.array([[
                initial_price - self._get_max_value(initial_price, terminal_time), 
                -10.0,  
                -10.0   
            ]]),
            max_value=np.array([[
                self._get_max_value(initial_price, terminal_time),  
                10.0,   
                10.0   
            ]]),
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=np.array([[initial_price, initial_fad, initial_filtered_fad]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )
        
        # Initialize filtering state
        if self.use_filtering:
            self._initialize_filtering()

    def _initialize_filtering(self):
        """Initialize the filtering components"""
        # Initialize filtered fad from state vector (3rd component)
        self.conditional_variance = np.zeros(self.num_trajectories)  
        self.innovation_process = np.zeros(self.num_trajectories) 
        
        # Pre-solve the Riccati equation for conditional variance
        self._solve_riccati_equation()

    def _solve_riccati_equation(self):
        """Solve the Riccati equation (47) with numerical stability checks"""
        def riccati_ode(t, P):

            return self.p**2 - P * (2*self.eta*(1-self.fads_proportion**2)) - (self.eta**2 * self.fads_proportion**2) * P**2
        
        time_grid = np.arange(0, self.terminal_time + self.step_size, self.step_size)
        
        sol = solve_ivp(
            riccati_ode, 
            [0, self.terminal_time], 
            [0], 
            t_eval=time_grid, 
            dense_output=True,
            rtol=1e-8,  
            atol=1e-10
        )
        
        max_variance = sol.y[0].max()
        if max_variance > 100:  # Sanity check
            print(f"Warning: Large conditional variance detected: {max_variance}")
       
        self.riccati_solution = sol.sol  

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None) -> np.ndarray:       
        time = state[:, TIME_INDEX]      
        self.current_time = time[0]  
        # print("Current time:", self.current_time)
        # if self.current_time == 0.001:
        #     print(state[:, INVENTORY_INDEX])
        
        normals = self.rng.normal(size=(self.num_trajectories, 2))
        Z = normals[:, 0]    
        B = normals[:, 1]    

        sqrt_dt = np.sqrt(self.step_size)
        dZ_inc = Z * sqrt_dt
        dB_inc = B * sqrt_dt

        previous_fad = self.current_state[:, 1].copy()

        if self.fads_proportion == 0.0:
            # --- No fad dynamics ---
            self.current_state[:, 1] = 0.0  
            dW_tilde = dZ_inc               
            drift_contribution = self.drift * self.step_size
        else:
            # --- OU update for fad ---
            exp_term = np.exp(-self.eta * self.step_size)
            std_term = np.sqrt((1 - np.exp(-2 * self.eta * self.step_size)) / (2 * self.eta))
            self.current_state[:, 1] = previous_fad * exp_term + std_term * B

            # Build correlated Brownian increment
            dW_tilde = self.p * dZ_inc + self.fads_proportion * dB_inc

            # Drift depends on fad
            drift_contribution = (self.drift - self.eta * self.fads_proportion * self.volatility * previous_fad) * self.step_size

        noise_contribution = self.volatility * dW_tilde
        self.current_state[:, 0] = self.current_state[:, 0] + drift_contribution + noise_contribution
                
        # --- Update filtered fad ---
        if self.use_filtering:
            if self.fads_proportion > 0.0:
                self._update_filtering(drift_contribution, noise_contribution)
            else:
                self.current_state[:, 2] = 0.0  # no fad to filter
        else:
            self.current_state[:, 2] = self.current_state[:, 1]

        return self.current_state

    def _update_filtering(self, expected_drift: np.ndarray, observed_noise: np.ndarray):
        """Update the filtering estimates based on observed midprice changes"""
        current_time = self.current_time
        
        # Get conditional variance from Riccati solution
        current_variance = self.riccati_solution(current_time)
        
        # Get current filtered fad estimate from state vector
        current_filtered_fad = self.current_state[:, 2]
        
        # Compute expected drift based on filtered estimate
        # π_t(h) = E[h(U_t) | F^S_t] = μ - ηqσ * Û_t
        expected_drift_filtered = (self.drift - self.eta * self.fads_proportion * self.volatility * current_filtered_fad) * self.step_size
        
        # Update innovation process (equation 43)
        # dI_t = dS_t - π_t(h)dt
        observed_price_change = expected_drift + observed_noise
        innovation_increment = observed_price_change - expected_drift_filtered
        
        self.innovation_process += innovation_increment
        
        # Update filtered fad estimate (equation 46) and store in state vector
        # dÛ_t = -η*Û_t*dt + σ^(-1)*(-ηq*P̂_t + q)*dI_t
        drift_term = -self.eta * current_filtered_fad * self.step_size
        innovation_gain = (1/self.volatility) * (-self.eta * self.fads_proportion * current_variance + self.fads_proportion)
        innovation_term = innovation_gain * innovation_increment
        
        # Update 3rd component of state vector (filtered fad)
        self.current_state[:, 2] = current_filtered_fad + drift_term + innovation_term
        
        # Update conditional variance (already solved via Riccati)
        self.conditional_variance = np.full(self.num_trajectories, current_variance)

    def get_filtered_state(self) -> np.ndarray:
        """Return the state available to partial info agent: [S_t, Û_t]"""
        return np.column_stack([
            self.current_state[:, 0],  
            self.current_state[:, 2]   
        ])
    
    def get_complete_info_state(self) -> np.ndarray:
        """Return the state available to complete info agent: [S_t, U_t]"""
        return np.column_stack([
            self.current_state[:, 0],  
            self.current_state[:, 1]   
        ])
    
    def get_true_state(self) -> np.ndarray:
        """Return the full state vector [S_t, U_t, Û_t] (for analysis)"""
        return self.current_state
    
    def get_filtered_fad(self) -> np.ndarray:
        """Return current filtered fad estimate Û_t"""
        return self.current_state[:, 2]
        
    def get_true_fad(self) -> np.ndarray:
        """Return current true fad U_t"""
        return self.current_state[:, 1]
    
    def get_conditional_variance(self) -> np.ndarray:
        """Return current conditional variance P̂_t = E[(U_t - Û_t)²]"""
        if not self.use_filtering:
            return np.zeros(self.num_trajectories)
        return self.conditional_variance
    
    def get_innovation_process(self) -> np.ndarray:
        """Return current innovation process I_t"""
        if not self.use_filtering:
            return np.zeros(self.num_trajectories)
        return self.innovation_process

    def _get_max_value(self, initial_price, terminal_time):
        # Account for both drift and volatility (including fad contribution)
        max_fad_impact = 4 * self.volatility * self.fads_proportion  # Approximate max fad contribution
        max_independent_noise = 4 * self.volatility * self.p * np.sqrt(terminal_time)
        return initial_price + max(self.drift * terminal_time, 0) + max_fad_impact + max_independent_noise


##########################################################
# A variant of the above model that without filtering
class ArithmeticBrownianMotionWithFadsMidpriceModelNoFilteredFad(MidpriceModel):
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
        self.drift = drift  # μ
        self.volatility = volatility  # σ 
        self.fads_proportion = fads_proportion # correlation parameter
        self.eta = eta  # mean reversion speed
        self.terminal_time = terminal_time
        
        # Derived parameters from paper
        self.p = np.sqrt(1 - self.fads_proportion**2)  # Independent noise coefficient
        
        super().__init__(
            min_value=np.array([[
                initial_price - self._get_max_value(initial_price, terminal_time),  # S_t min
                -10.0,  # U_t min (true fad)
                -10.0   # Û_t min (filtered fad)
            ]]),
            max_value=np.array([[
                self._get_max_value(initial_price, terminal_time),  # S_t max
                10.0,   # U_t max (true fad)
                10.0    # Û_t max (filtered fad)
            ]]),
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=np.array([[initial_price, initial_fad]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )
        

    
    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None) -> np.ndarray:       
        time = state[:, TIME_INDEX]      
        self.current_time = time[0]  # Use first element since all are the same
        
        # --- Sampling: use standard normals for B and Z, then scale by sqrt(dt) where needed ---
        # draw two independent standard normals per trajectory:
        normals = self.rng.normal(size=(self.num_trajectories, 2))
        Z = normals[:, 0]   
        B = normals[:, 1]   

        # OU exact update uses B directly (std_term already accounts for step)
        exp_term = np.exp(-self.eta * self.step_size)
        std_term = np.sqrt((1 - np.exp(-2 * self.eta * self.step_size)) / (2 * self.eta))

        previous_fad = self.current_state[:, 1].copy()
        # update true fad U_{t+dt}
        self.current_state[:, 1] = previous_fad * exp_term + std_term * B

        # Build correlated Brownian increment dW_tilde of order sqrt(dt):
        # dW_tilde = ( p * dZ + q * dB ), where dZ = Z * sqrt(dt), dB = B * sqrt(dt)
        sqrt_dt = np.sqrt(self.step_size)
        dZ_inc = Z * sqrt_dt
        dB_inc = B * sqrt_dt
        dW_tilde = self.p * dZ_inc + self.fads_proportion * dB_inc  # this is O(sqrt(dt))

        # --- Update midprice S_t (observable):
        # dS_t = h(U_t) * dt + c * dW_tilde, with h(U)=mu - eta*q*sigma*U
        drift_contribution = (self.drift - self.eta * self.fads_proportion * self.volatility * previous_fad) * self.step_size
        noise_contribution = self.volatility * dW_tilde  # already scaled by sqrt(dt)

        self.current_state[:, 0] = self.current_state[:, 0] + drift_contribution + noise_contribution
            
        return self.current_state

    def _get_max_value(self, initial_price, terminal_time):
        # we are assuming that max value is given related just to the brownian motion part without considering the fads
        return initial_price + 4 * self.volatility * terminal_time