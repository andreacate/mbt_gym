import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn as sns

from mbt_gym.agents.Agent import Agent
from mbt_gym.gym.TradingEnvironment import TradingEnvironment 
from mbt_gym.agents.SbAgent import SbAgent
from mbt_gym.gym.index_names import CASH_INDEX, INVENTORY_INDEX, ASSET_PRICE_INDEX, FADS_INDEX, FILTERED_FADS_INDEX
from mbt_gym.gym.helpers.generate_trajectory import generate_trajectory, generate_trajectory_rl, generate_trajectory_rl_fast


def plot_trajectory_extended(env: gym.Env, agent: Agent, seed: int = None):
    """
    Enhanced trajectory plotting that handles environments with additional state attributes
    from stochastic processes (beyond cash, inventory, asset_price).
    """
    timestamps = get_timestamps(env)
    observations, actions, rewards = generate_trajectory(env, agent, seed)
    action_dim = actions.shape[1]
    state_dim = observations.shape[1]
    
    # Calculate number of additional state dimensions beyond the basic 3 (cash, inventory, time)
    additional_dims = state_dim - 3
    
    # Determine subplot layout based on number of state dimensions
    if additional_dims <= 1:
        # Use original 2x2 layout for basic case
        fig, axes = plt.subplots(2, 2, figsize=(20, 10))
        axes = axes.flatten()
    else:
        # Dynamic layout: ensure we have enough subplots
        n_plots = 3 + additional_dims  # rewards, actions, basic states + additional states
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 5*n_rows))
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows > 1 else axes
    
    colors = ["r", "k", "b", "g", "m", "c", "y", "orange", "purple", "brown"]
    rewards = np.squeeze(rewards, axis=1)
    cum_rewards = np.cumsum(rewards, axis=-1)
    
    # Extract basic state components
    cash_holdings = observations[:, CASH_INDEX, :]
    inventory = observations[:, INVENTORY_INDEX, :]
    
    # Plot 1: Cumulative rewards
    ax_idx = 0
    axes[ax_idx].set_title("Cumulative Rewards")
    for i in range(env.num_trajectories):
        traj_label = f" trajectory {i}" if env.num_trajectories > 1 else ""
        axes[ax_idx].plot(timestamps[1:], cum_rewards[i, :], 
                         label=f"Cum Rewards{traj_label}",
                         alpha=(i + 1) / (env.num_trajectories + 1))
    if env.num_trajectories > 1:
        axes[ax_idx].legend()
    
    # Plot 2: Actions
    ax_idx = 1
    axes[ax_idx].set_title("Actions")
    for i in range(env.num_trajectories):
        traj_label = f" trajectory {i}" if env.num_trajectories > 1 else ""
        for j in range(action_dim):
            axes[ax_idx].plot(
                timestamps[0:-1],
                actions[i, j, :],
                label=f"Action {j}{traj_label}",
                color=colors[j % len(colors)],
                alpha=(i + 1) / (env.num_trajectories + 1),
            )
    axes[ax_idx].legend()
    
    # Plot 3: Basic agent state (inventory and cash)
    ax_idx = 2
    axes[ax_idx].set_title("Inventory and Cash Holdings")
    ax_cash = axes[ax_idx].twinx()
    
    for i in range(env.num_trajectories):
        traj_label = f" trajectory {i}" if env.num_trajectories > 1 else ""
        axes[ax_idx].plot(
            timestamps,
            inventory[i, :],
            label=f"Inventory{traj_label}",
            color="r",
            alpha=(i + 1) / (env.num_trajectories + 1),
        )
        ax_cash.plot(
            timestamps,
            cash_holdings[i, :],
            label=f"Cash{traj_label}",
            color="b",
            alpha=(i + 1) / (env.num_trajectories + 1),
        )
    
    axes[ax_idx].set_ylabel("Inventory", color="r")
    ax_cash.set_ylabel("Cash Holdings", color="b")
    axes[ax_idx].legend(loc='upper left')
    ax_cash.legend(loc='upper right')
    
    # print(env.stochastic_process_indices)
    # Plot additional state dimensions from stochastic processes
    if hasattr(env, 'stochastic_process_indices') and env.stochastic_process_indices:
        ax_idx = 3
        for process_name, (start_idx, end_idx) in env.stochastic_process_indices.items():
            if end_idx <= start_idx:
                continue  # Skip processes with no dimensions
            if ax_idx >= len(axes):
                break

            axes[ax_idx].set_title(f"{process_name.replace('_', ' ').title()}")

            # The first dimension uses the main axis, others use twinx
            twin_axes = [axes[ax_idx]]
            for dim in range(1, end_idx - start_idx):
                twin_axes.append(axes[ax_idx].twinx())
                # Offset the spine for visibility
                twin_axes[-1].spines["right"].set_position(("outward", 60 * dim))

            # Plot each dimension of this stochastic process
            for dim_idx in range(start_idx, end_idx):
                ax_dim = twin_axes[dim_idx - start_idx]
                for i in range(env.num_trajectories):
                    traj_label = f" traj {i}" if env.num_trajectories > 1 else ""
                    dim_label = f"Dim {dim_idx-start_idx}{traj_label}" if (end_idx - start_idx) > 1 else f"{process_name}{traj_label}"

                    ax_dim.plot(
                        timestamps,
                        observations[i, dim_idx, :],
                        label=dim_label,
                        color=colors[(dim_idx-start_idx) % len(colors)],
                        alpha=(i + 1) / (env.num_trajectories + 1),
                    )
                ax_dim.set_ylabel(dim_label)
                ax_dim.legend(loc='upper left' if dim_idx == start_idx else 'upper right')

            ax_idx += 1
    else:
        # Fallback: plot asset prices if ASSET_PRICE_INDEX exists and we have extra space
        if ax_idx < len(axes) and state_dim > 3:
            try:
                asset_prices = observations[:, ASSET_PRICE_INDEX, :]
                axes[ax_idx].set_title("Asset Prices")
                for i in range(env.num_trajectories):
                    traj_label = f" trajectory {i}" if env.num_trajectories > 1 else ""
                    axes[ax_idx].plot(timestamps, asset_prices[i, :], 
                                    label=f"Asset Price{traj_label}",
                                    alpha=(i + 1) / (env.num_trajectories + 1))
                if env.num_trajectories > 1:
                    axes[ax_idx].legend()
                ax_idx += 1
            except:
                # ASSET_PRICE_INDEX not defined or out of bounds
                pass

    # Hide unused subplots
    for i in range(ax_idx, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_trajectory(env: gym.Env, agent: Agent, seed: int = None):
    
    # assert env.num_trajectories == 1, "Plotting a trajectory can only be done when env.num_trajectories == 1."
    timestamps = get_timestamps(env)
    observations, actions, rewards = generate_trajectory(env, agent, seed)
    action_dim = actions.shape[1]
    colors = ["r", "k", "b", "g"]
    rewards = np.squeeze(rewards, axis=1)
    cum_rewards = np.cumsum(rewards, axis=-1)
    cash_holdings = observations[:, CASH_INDEX, :]
    inventory = observations[:, INVENTORY_INDEX, :]
    asset_prices = observations[:, ASSET_PRICE_INDEX, :]
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 10))
    ax3a = ax3.twinx()
    ax1.title.set_text("cum_rewards")
    ax2.title.set_text("asset_prices")
    ax3.title.set_text("inventory and cash holdings")
    ax4.title.set_text("Actions")
    for i in range(env.num_trajectories):
        traj_label = f" trajectory {i}" if env.num_trajectories > 1 else ""
        ax1.plot(timestamps[1:], cum_rewards[i, :])
        ax2.plot(timestamps, asset_prices[i, :])
        ax3.plot(
            timestamps,
            inventory[i, :],
            label=f"inventory" + traj_label,
            color="r",
            alpha=(i + 1) / (env.num_trajectories + 1),
        )
        ax3a.plot(
            timestamps,
            cash_holdings[i, :],
            label=f"cash holdings" + traj_label,
            color="b",
            alpha=(i + 1) / (env.num_trajectories + 1),
        )
        for j in range(action_dim):
            ax4.plot(
                timestamps[0:-1],
                actions[i, j, :],
                label=f"Action {j}" + traj_label,
                color=colors[j],
                alpha=(i + 1) / (env.num_trajectories + 1),
            )
    ax3.legend()
    ax4.legend()
    plt.show()


def plot_stable_baselines_actions(model, env):
    timestamps = get_timestamps(env)
    inventory_action_dict = {}
    price = 100
    cash = 100
    for inventory in [-3, -2, -1, 0, 1, 2, 3]:
        actions = model.predict([price, cash, inventory, 0], deterministic=True)[0].reshape((1, 2))
        for ts in timestamps[1:]:
            actions = np.append(
                actions, model.predict([price, cash, inventory, ts], deterministic=True)[0].reshape((1, 2)), axis=0
            )
        inventory_action_dict[inventory] = actions
    for inventory in [-3, -2, -1, 0, 1, 2, 3]:
        plt.plot(np.array(inventory_action_dict[inventory]).T[0], label=inventory)
    plt.legend()
    plt.show()
    for inventory in [-3, -2, -1, 0, 1, 2, 3]:
        plt.plot(np.array(inventory_action_dict[inventory]).T[1], label=inventory)
    plt.legend()
    plt.show()


def plot_pnl(rewards, symmetric_rewards=None):
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    if symmetric_rewards is not None:
        sns.histplot(symmetric_rewards, label="Rewards of symmetric strategy", stat="density", bins=50, ax=ax)
    sns.histplot(rewards, label="Rewards", color="red", stat="density", bins=50, ax=ax)
    ax.legend()
    plt.close()
    return fig


def generate_results_table_and_hist(vec_env: TradingEnvironment, agent: Agent, n_episodes: int = 1000):
    assert vec_env.num_trajectories > 1, "To generate a results table and hist, vec_env must roll out > 1 trajectory."
    observations, actions, rewards = generate_trajectory(vec_env, agent)
    total_rewards = rewards.sum(axis=-1).reshape(-1)
    terminal_inventories = observations[:, INVENTORY_INDEX, -1]
    half_spreads = actions.mean(axis=(-1, -2))

    rows = ["Inventory"]
    columns = ["Mean spread", "Mean PnL", "Std PnL", "Mean terminal inventory", "Std terminal inventory"]
    results = pd.DataFrame(index=rows, columns=columns)
    results.loc[:, "Mean spread"] = 2 * np.mean(half_spreads)
    results.loc["Inventory", "Mean PnL"] = np.mean(total_rewards)
    results.loc["Inventory", "Std PnL"] = np.std(total_rewards)
    results.loc["Inventory", "Mean terminal inventory"] = np.mean(terminal_inventories)
    results.loc["Inventory", "Std terminal inventory"] = np.std(terminal_inventories)
    fig = plot_pnl(total_rewards)
    return results, fig, total_rewards


# def get_timestamps(env: gym.Env):
#     return np.linspace(0, env.terminal_time, env.n_steps + 1)

def get_timestamps(env: gym.Env):
    # Check if it's a wrapper and try to access the wrapped env
    if hasattr(env, 'env') and hasattr(env.env, 'terminal_time'):
        # For a single wrapper
        terminal_time = env.env.terminal_time
        n_steps = env.env.n_steps
    elif hasattr(env, 'trading_env') and hasattr(env.trading_env, 'terminal_time'):
        # For StableBaselinesTradingEnvironment
        terminal_time = env.trading_env.terminal_time
        n_steps = env.trading_env.n_steps
    elif hasattr(env, 'terminal_time'):
        # Direct access for unwrapped environment
        terminal_time = env.terminal_time
        n_steps = env.n_steps
    else:
        # Fallback if terminal_time isn't available
        terminal_time = None
        n_steps = env.n_steps if hasattr(env, 'n_steps') else 1000  # Default fallback
    
    if not terminal_time:
        return np.linspace(0, 1, n_steps + 1)
    else:
        return np.linspace(0, terminal_time, n_steps + 1)


def plot_rl_trajectory_extended(env: gym.Env, agent: SbAgent, use_normalized_agent: bool = False, seed: int = None):
    """
    Enhanced trajectory plotting specifically for RL agents that includes
    policy state information and neural network activations.
    """

    print("Generating RL trajectory plot...")
    print("env.observation_space:", env.observation_space)
    print("env.action_space:", env.action_space)
    #print("env.normalise_observation_space:", env.normalise_observation_space)
    #print("env.normalise_action_space:", env.normalise_action_space)
    # Generate trajectory data
    timestamps = get_timestamps(env)
    observations, actions, rewards = generate_trajectory_rl(env, agent, use_normalized_agent=use_normalized_agent, seed=seed)
    action_dim = actions.shape[1]
    state_dim = observations.shape[1]
    
    # Create figure with additional subplot for RL policy information
    n_plots = 4  # Rewards, actions, states, policy info
    if hasattr(env, 'stochastic_process_indices') and env.stochastic_process_indices:
        n_plots += len(env.stochastic_process_indices)
    
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 5*n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 else axes
    
    colors = ["r", "k", "b", "g", "m", "c", "y", "orange", "purple", "brown"]
    rewards = np.squeeze(rewards, axis=1)
    cum_rewards = np.cumsum(rewards, axis=-1)
    
    # Extract basic state components
    cash_holdings = observations[:, CASH_INDEX, :] if CASH_INDEX < state_dim else None
    inventory = observations[:, INVENTORY_INDEX, :] if INVENTORY_INDEX < state_dim else None
    asset_prices = observations[:, ASSET_PRICE_INDEX, :] if ASSET_PRICE_INDEX < state_dim else None
    
    # Plot 1: Cumulative rewards
    ax_idx = 0
    axes[ax_idx].set_title("Cumulative Rewards")
    for i in range(env.num_trajectories):
        traj_label = f" trajectory {i}" if env.num_trajectories > 1 else ""
        axes[ax_idx].plot(timestamps[1:], cum_rewards[i, :], 
                         label=f"Cum Rewards{traj_label}",
                         alpha=(i + 1) / (env.num_trajectories + 1))
    if env.num_trajectories > 1:
        axes[ax_idx].legend()
    
    # Plot 2: Actions
    ax_idx = 1
    axes[ax_idx].set_title("Actions")
    for i in range(env.num_trajectories):
        traj_label = f" trajectory {i}" if env.num_trajectories > 1 else ""
        for j in range(action_dim):
            axes[ax_idx].plot(
                timestamps[0:-1],
                actions[i, j, :],
                label=f"Action {j}{traj_label}",
                color=colors[j % len(colors)],
                alpha=(i + 1) / (env.num_trajectories + 1),
            )
    axes[ax_idx].legend()
    
    # Plot 3: Basic agent state (inventory and cash)
    ax_idx = 2
    axes[ax_idx].set_title("Inventory and Cash Holdings")
    
    # If we have inventory data
    if inventory is not None:
        for i in range(env.num_trajectories):
            traj_label = f" trajectory {i}" if env.num_trajectories > 1 else ""
            axes[ax_idx].plot(
                timestamps,
                inventory[i, :],
                label=f"Inventory{traj_label}",
                color="r",
                alpha=(i + 1) / (env.num_trajectories + 1),
            )
        axes[ax_idx].set_ylabel("Inventory", color="r")
        axes[ax_idx].legend(loc='upper left')
    
    # If we have cash data, add to the same plot with a second y-axis
    if cash_holdings is not None:
        ax_cash = axes[ax_idx].twinx()
        for i in range(env.num_trajectories):
            traj_label = f" trajectory {i}" if env.num_trajectories > 1 else ""
            ax_cash.plot(
                timestamps,
                cash_holdings[i, :],
                label=f"Cash{traj_label}",
                color="b",
                alpha=(i + 1) / (env.num_trajectories + 1),
            )
        ax_cash.set_ylabel("Cash Holdings", color="b")
        ax_cash.legend(loc='upper right')
    
    # Plot 4: Asset prices (if available)
    ax_idx = 3
    if asset_prices is not None:
        axes[ax_idx].set_title("Asset Prices")
        for i in range(env.num_trajectories):
            traj_label = f" trajectory {i}" if env.num_trajectories > 1 else ""
            axes[ax_idx].plot(
                timestamps,
                asset_prices[i, :],
                label=f"Asset Price{traj_label}",
                alpha=(i + 1) / (env.num_trajectories + 1)
            )
        if env.num_trajectories > 1:
            axes[ax_idx].legend()
    
    # Plot 5: RL Policy Information
    ax_idx = 4
    axes[ax_idx].set_title("RL Policy Network Weights")
    
    # Get policy state dictionary
    policy_state = agent.get_state()
    
    # Visualize neural network parameters
    if "mlp_extractor.policy_net.0.weight" in policy_state:
        # Extract first layer weights for visualization
        weights = policy_state["mlp_extractor.policy_net.0.weight"].detach().numpy()
        im = axes[ax_idx].imshow(weights, cmap='viridis', aspect='auto')
        axes[ax_idx].set_xlabel("Input features")
        axes[ax_idx].set_ylabel("Neurons")
        plt.colorbar(im, ax=axes[ax_idx], label="Weight value")
    else:
        # Fallback if weights not accessible in expected format
        axes[ax_idx].text(0.5, 0.5, "Policy network structure unavailable", 
                        ha='center', va='center', transform=axes[ax_idx].transAxes)
    
    # Plot additional state dimensions from stochastic processes
    if hasattr(env, 'stochastic_process_indices') and env.stochastic_process_indices:
        ax_idx = 5
        for process_name, (start_idx, end_idx) in env.stochastic_process_indices.items():
            if end_idx <= start_idx or ax_idx >= len(axes):
                continue  # Skip processes with no dimensions or if we run out of subplots
            axes[ax_idx].set_title(f"{process_name.replace('_', ' ').title()}")
            
            # Plot each dimension of this stochastic process
            twin_axes = [axes[ax_idx]]
            for dim in range(1, end_idx - start_idx):
                if dim > 0:  # Create twin axis for additional dimensions
                    twin_axes.append(axes[ax_idx].twinx())
                    twin_axes[-1].spines["right"].set_position(("outward", 60 * dim))
                    
            for dim_idx in range(start_idx, end_idx):
                ax_dim = twin_axes[dim_idx - start_idx]
                for i in range(env.num_trajectories):
                    traj_label = f" traj {i}" if env.num_trajectories > 1 else ""
                    dim_label = f"Dim {dim_idx-start_idx}{traj_label}" if (end_idx - start_idx) > 1 else f"{process_name}{traj_label}"
                    
                    ax_dim.plot(
                        timestamps,
                        observations[i, dim_idx, :],
                        label=dim_label,
                        color=colors[(dim_idx-start_idx) % len(colors)],
                        alpha=(i + 1) / (env.num_trajectories + 1),
                    )
                ax_dim.set_ylabel(dim_label)
                ax_dim.legend(loc='upper left' if dim_idx == start_idx else 'upper right')
                
            ax_idx += 1
    
    # Add an additional subplot for model architecture visualization
    if ax_idx < len(axes):
        axes[ax_idx].set_title("RL Model Architecture")
        try:
            # Create a text description of the model architecture
            model_info = str(agent.model.policy).split('\n')
            y_pos = 0.95
            for line in model_info:
                axes[ax_idx].text(0.05, y_pos, line, fontsize=9, va='top', transform=axes[ax_idx].transAxes)
                y_pos -= 0.05
            axes[ax_idx].axis('off')
            ax_idx += 1
        except:
            axes[ax_idx].text(0.5, 0.5, "Model architecture unavailable", 
                            ha='center', va='center', transform=axes[ax_idx].transAxes)
            ax_idx += 1
    
    # Hide unused subplots
    for i in range(ax_idx, len(axes)):
        axes[i].set_visible(False)
        
    plt.tight_layout()
    plt.show()
    return fig


def generate_results_table_and_hist_rl(vec_env: TradingEnvironment, rl_agent: SbAgent, use_normalized_agent: bool = False, n_episodes: int = 1000):
    """
    Generate results table and histogram for RL agents that require observation transformation.
    
    Args:
        vec_env: Vectorized trading environment with multiple trajectories
        rl_agent: RL agent from stable baselines
        n_episodes: Number of episodes for statistics (not used directly, uses vec_env.num_trajectories)
        
    Returns:
        results: DataFrame with performance statistics
        fig: Figure with PnL histogram
        total_rewards: Array of total rewards for each trajectory
    """
    assert vec_env.num_trajectories > 1, "To generate a results table and hist, vec_env must roll out > 1 trajectory."
    
    # Use the RL-specific trajectory generator
    observations, actions, rewards = generate_trajectory_rl(vec_env, rl_agent, use_normalized_agent=use_normalized_agent)
    
    total_rewards = rewards.sum(axis=-1).reshape(-1)
    terminal_inventories = observations[:, INVENTORY_INDEX, -1]
    half_spreads = actions.mean(axis=(-1, -2))

    rows = ["RL Agent"]
    columns = ["Mean spread", "Mean PnL", "Std PnL", "Mean terminal inventory", "Std terminal inventory"]
    results = pd.DataFrame(index=rows, columns=columns)
    
    results.loc[:, "Mean spread"] = 2 * np.mean(half_spreads)
    results.loc["RL Agent", "Mean PnL"] = np.mean(total_rewards)
    results.loc["RL Agent", "Std PnL"] = np.std(total_rewards)
    results.loc["RL Agent", "Mean terminal inventory"] = np.mean(terminal_inventories)  
    results.loc["RL Agent", "Std terminal inventory"] = np.std(terminal_inventories)
    
    # Create PnL histogram
    fig = plot_pnl(total_rewards)
    
    return results, fig, total_rewards

def generate_results_table_and_hist_rl_fast(vec_env: TradingEnvironment, rl_agent: SbAgent, use_normalized_agent: bool = False, n_episodes: int = 1000):
    
    total_rewards, terminal_inventory, mean_spread = generate_trajectory_rl_fast(
        vec_env, rl_agent, use_normalized_agent=use_normalized_agent)

    results = pd.DataFrame({
        "Mean spread": [np.mean(mean_spread)],
        "Mean PnL": [np.mean(total_rewards)],
        "Std PnL": [np.std(total_rewards)],
        "Mean terminal inventory": [np.mean(terminal_inventory)],
        "Std terminal inventory": [np.std(terminal_inventory)]
    }, index=["RL Agent"])

    fig = plot_pnl(total_rewards)

    return results, fig, total_rewards
