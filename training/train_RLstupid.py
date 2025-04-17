import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from core.game_manager import GameManager
from core.game_state import GameState
from core.risk_map import simple_adjacency as adjacency
from agents.ppo_agent import PPOAgent
from agents.stupid_bot import StupidBot

def calculate_reward(state, winner, turn_count, player_id=1):
    if winner is not None:
        # Large reward for winning, large penalty for losing
        return 10.0 if winner == player_id else -10.0
    
    # Small reward for each territory owned above initial 5
    territories_owned = len(state.get_owned_territories(player_id))
    territory_reward = 0.2 * (territories_owned - 5)
    
    # Small penalty for taking too many turns
    turn_penalty = -0.01 * turn_count
    
    return territory_reward + turn_penalty

def train_ppo_against_stupid(
    total_episodes=1000,
    update_every=1,
    state_dim=20,
    action_dim=10,
    eval_every=500
):
    
    ppo_agent = PPOAgent(
        player_id=1,
        state_dim=state_dim,
        action_dim=action_dim,
        lr=0.0001,
        gamma=0.99,
        epsilon=0.2,
        entropy_coef=0.05
    )
    stupid_bot = StupidBot(player_id=2)

    win_history = []
    reward_history = []
    policy_loss_history = []
    value_loss_history = []
    entropy_history = []

    for episode in range(1, total_episodes + 1):
            
        state = GameState(adjacency=adjacency)
        for i in range(5):
            state.set_ownership(i, 1)
            state.set_troops(i, 1)
        for i in range(5, 10):
            state.set_ownership(i, 2)
            state.set_troops(i, 1)

        game = GameManager(state, agent1=ppo_agent, agent2=stupid_bot)

        turn_count = 0
        old_state = None
        action = None

        while not game.is_game_over():
            player = game.current_player

            if player == 1:
                if old_state is not None and action is not None:
                    reward = calculate_reward(state, None, turn_count)
                    ppo_agent.remember(old_state, action, reward)

                old_state = np.array(state.get_state_vector())
                action, _, _ = ppo_agent.get_action(old_state)

            if episode <= 5 and turn_count % 10 == 0:  # Debug first 5 episodes
                print(f"Episode {episode}, Turn {turn_count}")
                print(f"Current state:\n{state}")

            # Add bonus troops based on territory control (only difference from greedy)
            if player == 1:
                bonus_troops = max(0, len(state.get_owned_territories(1)) - 5)
                state.remaining_troops[1] += bonus_troops
            elif player == 2:
                bonus_troops = max(0, len(state.get_owned_territories(2)) - 5)
                state.remaining_troops[2] += bonus_troops

            game.run_turn()
            turn_count += 1

            # Safety check for very long games
            if turn_count > 1000:
                print(f"Game too long! Ending at turn {turn_count}")
                break

        winner = game.get_winner()
        final_reward = calculate_reward(state, winner, turn_count)

        if old_state is not None and action is not None:
            ppo_agent.remember(old_state, action, final_reward)

        win_history.append(winner)
        reward_history.append(final_reward)

        if episode % update_every == 0:
            # Calculate statistics over different time windows
            win_rate = np.mean([1 if w == 1 else 0 for w in win_history[-update_every:]])
            avg_reward = np.mean(reward_history[-update_every:])

            recent_win_rate = np.mean([1 if w == 1 else 0 for w in win_history[-update_every:]])
            long_term_win_rate = np.mean([1 if w == 1 else 0 for w in win_history[-update_every*5:]])
            
            recent_reward = np.mean(reward_history[-update_every:])
            long_term_reward = np.mean(reward_history[-update_every*5:])
            
            # Get losses from the update
            policy_loss, value_loss, entropy = ppo_agent.update()
            policy_loss_history.append(policy_loss)
            value_loss_history.append(value_loss)
            entropy_history.append(entropy)
            
            print(f"Episode {episode:5d} | Winrate: {win_rate:.2f} | Reward: {avg_reward:.2f} | Policy Loss: {policy_loss:.4f} | Value Loss: {value_loss:.4f} | Entropy: {entropy:.4f}")

            ppo_agent.clear_memory()

    print("Training complete!")

    plt.figure(figsize=(15, 12))

    # Win rate comparison (recent vs long-term)
    plt.subplot(4, 1, 1)
    episode_axis = np.arange(1, total_episodes + 1)
    
    # Calculate win rates with smaller windows for more detail
    window_small = 20  # Show more recent fluctuations
    window_large = 100  # Show longer-term trend
    
    recent_winrate = []
    long_term_winrate = []
    
    for i in range(window_large, len(win_history) + 1):
        recent_winrate.append(np.mean([1 if w == 1 else 0 for w in win_history[i-window_small:i]]))
        long_term_winrate.append(np.mean([1 if w == 1 else 0 for w in win_history[i-window_large:i]]))
    
    plt.plot(episode_axis[window_large-1:], recent_winrate, label=f'Recent Win Rate ({window_small} episodes)', alpha=0.7)
    plt.plot(episode_axis[window_large-1:], long_term_winrate, label=f'Long-term Win Rate ({window_large} episodes)', linewidth=2)
    plt.title('Win Rate Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    plt.grid(True)
    plt.legend()

    # Reward comparison (recent vs long-term)
    plt.subplot(4, 1, 2)
    recent_reward = np.convolve(reward_history, np.ones(update_every)/update_every, mode='valid')
    long_term_reward = np.convolve(reward_history, np.ones(update_every*5)/(update_every*5), mode='valid')
    plt.plot(episode_axis[:len(recent_reward)], recent_reward, label='Recent Reward')
    plt.plot(episode_axis[:len(long_term_reward)], long_term_reward, label='Long-term Reward')
    plt.title('Reward Comparison')
    plt.grid(True)
    plt.legend()

    # Loss trends
    plt.subplot(4, 1, 3)
    update_axis = np.arange(update_every, total_episodes + 1, update_every)
    plt.plot(update_axis, policy_loss_history, label='Policy Loss')
    plt.plot(update_axis, value_loss_history, label='Value Loss')
    plt.title('Loss Trends')
    plt.grid(True)
    plt.legend()

    # Entropy and Win Rate Correlation
    plt.subplot(4, 1, 4)
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.plot(update_axis, entropy_history, 'g-', label='Entropy')
    ax2.plot(update_axis, [np.mean([1 if w == 1 else 0 for w in win_history[max(0, i-update_every):i]]) 
                          for i in range(update_every, total_episodes + 1, update_every)], 
             'b-', label='Win Rate')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Entropy', color='g')
    ax2.set_ylabel('Win Rate', color='b')
    plt.title('Entropy vs Win Rate')
    plt.grid(True)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    plt.show()

    return ppo_agent, win_history, reward_history

if __name__ == "__main__":
    train_ppo_against_stupid(total_episodes=5000) 