import numpy as np
from agents.ppo_agent import PPOAgent
from agents.greedy_bot import GreedyBot
from core.game_state import GameState
from core.game_manager import GameManager
from core.risk_map import adjacency
import matplotlib.pyplot as plt

def calculate_reward(state, winner, turn_count, player_id=1):
    if winner is not None:
        terminal_reward = 5.0 if winner == player_id else -5.0
    else:
        terminal_reward = 0

    territories_owned = len(state.get_owned_territories(player_id))
    territory_reward = 0.1 * (territories_owned - 5)

    total_reward = terminal_reward + territory_reward
    return total_reward

def train_ppo_against_greedy(
    total_episodes=1000,
    update_every=100,
    state_dim=20,
    action_dim=10,
    eval_every=500
):
    
    ppo_agent = PPOAgent(
        player_id=1,
        state_dim=state_dim,
        action_dim=action_dim,
        lr=0.00005,
        gamma=0.99,
        epsilon=0.1,
        entropy_coef=0.02
    )
    greedy_bot = GreedyBot(player_id=2)

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

        game = GameManager(state, agent1=ppo_agent, agent2=greedy_bot)

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
            win_rate = np.mean([1 if w == 1 else 0 for w in win_history[-update_every:]])
            avg_reward = np.mean(reward_history[-update_every:])
            
            # Get losses from the update
            policy_loss, value_loss, entropy = ppo_agent.update()
            policy_loss_history.append(policy_loss)
            value_loss_history.append(value_loss)
            entropy_history.append(entropy)
            
            print(f"Episode {episode:5d} | Winrate: {win_rate:.2f} | Reward: {avg_reward:.2f} | Policy Loss: {policy_loss:.4f} | Value Loss: {value_loss:.4f} | Entropy: {entropy:.4f}")

            ppo_agent.clear_memory()

    print("Training complete!")

    plt.figure(figsize=(15, 10))

    plt.subplot(3, 1, 1)
    episode_axis = np.arange(1, total_episodes + 1)
    rolling_winrate = np.convolve(
        [1 if w == 1 else 0 for w in win_history], np.ones(1)/1, mode='valid'
    )
    plt.plot(episode_axis[:len(rolling_winrate)], rolling_winrate, label='Training win rate (rolling avg)')

    plt.subplot(3, 1, 2)
    rolling_reward = np.convolve(reward_history, np.ones(100)/100, mode='valid')
    plt.plot(episode_axis[:len(rolling_reward)], rolling_reward, label='Average Reward')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 3)
    update_axis = np.arange(update_every, total_episodes + 1, update_every)
    plt.plot(update_axis, policy_loss_history, label='Policy Loss')
    plt.plot(update_axis, value_loss_history, label='Value Loss')
    plt.plot(update_axis, entropy_history, label='Entropy')
    plt.xlabel("Episode")
    plt.ylabel("Loss/Entropy")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    return ppo_agent, win_history, reward_history

if __name__ == "__main__":
    train_ppo_against_greedy(total_episodes=5000)
