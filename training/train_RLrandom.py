import numpy as np
from agents.ppo_agent import PPOAgent
from agents.random_bot import RandomBot
from core.game_state import GameState
from core.game_manager import GameManager
from core.risk_map import adjacency
import matplotlib.pyplot as plt


def train_ppo_against_random(
    total_episodes=10000,
    update_every=50,
    state_dim=20,     # 10 territories * 2 features (owner, troops)
    action_dim=10     # max number of possible choices per phase
):
    # Initialize agents
    ppo_agent = PPOAgent(player_id=1, state_dim=state_dim, action_dim=action_dim)
    random_bot = RandomBot(player_id=2)

    win_history = []
    reward_history = []
    turn_history = []

    for episode in range(1, total_episodes + 1):
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
        # Run game until it ends
        while not game.is_game_over():
            current_state = state.get_state_vector()
            player = game.current_player
            if player == 1:
                old_state = np.array(state.get_state_vector())
                action, _, _ = ppo_agent.get_action(old_state)

            game.run_turn()
            turn_count += 1

        # Determine reward based on win/loss
        winner = game.get_winner()
        reward = 1 if winner == 1 else -1

        # Store trajectory for PPO
        if old_state is not None and action is not None:
            ppo_agent.remember(old_state, action, reward)

        win_history.append(winner)
        reward_history.append(reward)
        turn_history.append(turn_count)


        # reports every n turns
        if episode % update_every == 0:
            win_rate = np.mean([1 if w == 1 else 0 for w in win_history[-update_every:]])
            avg_turns = np.mean(turn_history[-update_every:])
            print(f"Episode {episode:5d} | PPO win rate: {win_rate:.2f} | Avg turns: {avg_turns:.1f}")

            ppo_agent.update()
            ppo_agent.clear_memory()

    print("Training complete!")

    # Plot results
    episode_axis = np.arange(1, total_episodes + 1)
    rolling_winrate = np.convolve(
        [1 if w == 1 else 0 for w in win_history], np.ones(100)/100, mode='valid'
    )

    plt.figure(figsize=(10, 4))
    plt.plot(episode_axis[:len(rolling_winrate)], rolling_winrate, label='PPO win rate (rolling avg)')
    plt.xlabel("Episode")
    plt.ylabel("Win Rate")
    plt.title("PPO Agent Performance vs RandomBot")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return ppo_agent, win_history, reward_history


if __name__ == "__main__":
    train_ppo_against_random(total_episodes=10000)