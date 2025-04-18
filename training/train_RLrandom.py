import numpy as np
from agents.ppo_agent import PPOAgent
from agents.random_bot import RandomBot
from core.game_state import GameState
from core.game_manager import GameManager
from core.risk_map import adjacency
import matplotlib.pyplot as plt

def calculate_reward(state, winner, turn_count, player_id=1):
    # Terminal reward
    if winner is not None:
        terminal_reward = 5.0 if winner == player_id else -5.0
    else:
        terminal_reward = 0
        
    # Territory control reward
    territories_owned = len(state.get_owned_territories(player_id))
    territory_reward = 0.1 * (territories_owned - 5)  # -5 is initial number
    
    '''
    # Troop advantage reward
    player_troops = sum(state.get_troops(t) for t in state.get_owned_territories(player_id))
    opponent_troops = sum(state.get_troops(t) for t in state.get_owned_territories(3-player_id))
    troop_advantage = 0.05 * (player_troops - opponent_troops)
    
    # Turn penalty to encourage faster wins
    turn_penalty = -0.01 * turn_count
    '''
    
    total_reward = terminal_reward + territory_reward #+ troop_advantage + turn_penalty
    return total_reward

def train_ppo_against_random(
    total_episodes=50000,
    update_every=100,  # Changed back to 100
    state_dim=20,     # 10 territories * 2 features (owner, troops)
    action_dim=10,    # max number of possible choices per phase
    eval_every=500    # Evaluate performance periodically
):
    print("Starting training...")  # Debug print
    
    # Initialize agents
    ppo_agent = PPOAgent(
        player_id=1, 
        state_dim=state_dim, 
        action_dim=action_dim,
        lr=0.00005,  # Further reduced learning rate
        gamma=0.99,  # Back to standard gamma
        epsilon=0.1,
        entropy_coef=0.02  # Increased entropy for more exploration
    )
    random_bot = RandomBot(player_id=2)

    win_history = []
    reward_history = []
    turn_history = []
    eval_wins = []

    for episode in range(1, total_episodes + 1):
        if episode % 100 == 0:  # Debug print every 100 episodes regardless of update_every
            print(f"Starting episode {episode}")
            
        # Reset game
        state = GameState(adjacency=adjacency)
        for i in range(5):
            state.set_ownership(i, 1)
            state.set_troops(i, 1)
        for i in range(5, 10):
            state.set_ownership(i, 2)
            state.set_troops(i, 1)

        game = GameManager(state, agent1=ppo_agent, agent2=random_bot)

        turn_count = 0
        old_state = None
        action = None
        
        # Run game until it ends
        while not game.is_game_over():
            current_state = state.get_state_vector()
            player = game.current_player
            
            if player == 1:
                # Store previous state-action pair's reward if exists
                if old_state is not None and action is not None:
                    reward = calculate_reward(state, None, turn_count)
                    ppo_agent.remember(old_state, action, reward)
                
                # Get new action
                old_state = np.array(state.get_state_vector())
                action, _, _ = ppo_agent.get_action(old_state)

            game.run_turn()
            turn_count += 1

        # Final reward based on game outcome
        winner = game.get_winner()
        final_reward = calculate_reward(state, winner, turn_count)
        
        # Store final trajectory for PPO
        if old_state is not None and action is not None:
            ppo_agent.remember(old_state, action, final_reward)

        win_history.append(winner)
        reward_history.append(final_reward)
        turn_history.append(turn_count)

        if episode % 100 == 0:  # Debug print game completion
            print(f"Episode {episode} completed in {turn_count} turns. Winner: {winner}")

        # Update every n episodes
        if episode % update_every == 0:
            win_rate = np.mean([1 if w == 1 else 0 for w in win_history[-update_every:]])
            avg_reward = np.mean(reward_history[-update_every:])
            avg_turns = np.mean(turn_history[-update_every:])
            print(f"Episode {episode:5d} | Win rate: {win_rate:.2f} | Reward: {avg_reward:.2f} | Turns: {avg_turns:.1f}")
            
            ppo_agent.update()
            ppo_agent.clear_memory()

    print("Training complete!")

    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot win rate
    plt.subplot(3, 1, 1)
    episode_axis = np.arange(1, total_episodes + 1)
    rolling_winrate = np.convolve(
        [1 if w == 1 else 0 for w in win_history], np.ones(1000)/1000, mode='valid'
    )
    plt.plot(episode_axis[:len(rolling_winrate)], rolling_winrate, label='Training win rate (rolling avg)')
    
    # Plot rewards
    plt.subplot(3, 1, 2)
    rolling_reward = np.convolve(reward_history, np.ones(100)/100, mode='valid')
    plt.plot(episode_axis[:len(rolling_reward)], rolling_reward, label='Average Reward')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()
    
    # Plot average turns
    plt.subplot(3, 1, 3)
    rolling_turns = np.convolve(turn_history, np.ones(100)/100, mode='valid')
    plt.plot(episode_axis[:len(rolling_turns)], rolling_turns, label='Average Game Length')
    plt.xlabel("Episode")
    plt.ylabel("Turns")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    return ppo_agent, win_history, reward_history

def evaluate_agent(agent, num_games):
    """Evaluate agent performance without updating."""
    wins = 0
    random_bot = RandomBot(player_id=2)
    
    for _ in range(num_games):
        state = GameState(adjacency=adjacency)
        for i in range(5):
            state.set_ownership(i, 1)
            state.set_troops(i, 1)
        for i in range(5, 10):
            state.set_ownership(i, 2)
            state.set_troops(i, 1)
            
        game = GameManager(state, agent1=agent, agent2=random_bot)
        
        while not game.is_game_over():
            game.run_turn()
            
        if game.get_winner() == 1:
            wins += 1
            
    return wins / num_games

if __name__ == "__main__":
    train_ppo_against_random(total_episodes=50000)