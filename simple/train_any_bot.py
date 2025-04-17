import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from simple.game_state import SimpleGameState
from simple.ppo_agent import SimplePPOAgent
import matplotlib.pyplot as plt

# Define action space
attack_pairs = [
    (0, 1), (0, 2), (1, 0), (1, 3), (1, 4),
    (2, 0), (2, 4), (2, 5), (3, 1), (3, 6),
    (4, 1), (4, 2), (4, 7), (5, 2), (5, 8),
    (6, 3), (6, 7), (7, 4), (7, 6), (7, 8), (7, 9),
    (8, 5), (8, 7), (8, 9), (9, 7), (9, 8),
    ('pass',)  # Pass action
]

def get_frontline_territories(game_state, player):
    """Get territories that are adjacent to enemy territories"""
    frontline = set()
    for territory in game_state.get_owned_territories(player):
        for adj in game_state.adjacency[territory]:
            if game_state.get_owner(adj) != player:
                frontline.add(territory)
                break
    return frontline

def auto_deploy(game_state, player):
    """Automatically deploy troops across all territories"""
    # Get deployment troops based on territory count
    deployment_troops = game_state.get_deployment_troops(player)
    game_state.add_deployment_troops(player, deployment_troops)
    
    # Get all owned territories
    owned_territories = game_state.get_owned_territories(player)
    
    # Calculate troops per territory (minimum 1)
    base_troops = max(1, deployment_troops // len(owned_territories))
    extra_troops = deployment_troops % len(owned_territories)
    
    # First, ensure each territory has at least 1 troop
    for territory in owned_territories:
        if game_state.get_troops(territory) < 1:
            game_state.deploy(player, territory, 1)
            deployment_troops -= 1
    
    # Then distribute remaining troops to frontline territories
    frontline = get_frontline_territories(game_state, player)
    if not frontline:
        frontline = owned_territories
    
    # Distribute remaining troops
    for i, territory in enumerate(frontline):
        if deployment_troops <= 0:
            break
        troops = min(base_troops + (1 if i < extra_troops else 0), deployment_troops)
        game_state.deploy(player, territory, troops)
        deployment_troops -= troops

def auto_fortify(game_state, player):
    """Move troops from non-frontline territories to frontline territories"""
    frontline = get_frontline_territories(game_state, player)
    non_frontline = [t for t in game_state.get_owned_territories(player) if t not in frontline]
    
    for territory in non_frontline:
        troops = game_state.get_troops(territory)
        if troops > 1:  # Leave at least 1 troop
            # Find adjacent frontline territory
            for adj in game_state.adjacency[territory]:
                if adj in frontline:
                    # Move all but 1 troop to the frontline territory
                    move_troops = troops - 1
                    game_state.state[territory, 1] -= move_troops
                    game_state.state[adj, 1] += move_troops
                    break

def train_ppo_against_bot(opponent_bot, num_episodes=1000, update_every=10):
    """
    Train PPO agent against any bot implementation
    
    Args:
        opponent_bot: The bot to train against (must implement get_action method)
        num_episodes: Number of training episodes
        update_every: How often to update the PPO agent
    """
    # Initialize game state and agents
    game_state = SimpleGameState()
    ppo_agent = SimplePPOAgent(state_dim=120, action_dim=len(attack_pairs))
    
    # Training metrics
    win_history = []
    reward_history = []
    turn_history = []
    policy_losses = []
    value_losses = []
    entropies = []
    territory_counts = []
    
    for episode in range(num_episodes):
        # Reset game state
        game_state = SimpleGameState()
        done = False
        total_reward = 0
        turns = 0
        
        # Debug print for first 5 episodes
        if episode < 5:
            print(f"\nEpisode {episode + 1} initial state:")
            print(game_state)
        
        while not done:
            # PPO agent's turn
            if game_state.current_player == 1:
                # Auto deploy and fortify
                auto_deploy(game_state, 1)
                auto_fortify(game_state, 1)
                
                # Get current state
                current_state = game_state.get_state_vector()
                print(f"State vector shape: {current_state.shape}")  # Debug print
                action, action_prob = ppo_agent.get_action(game_state)
                
                # Execute action
                if action < len(attack_pairs) - 1:  # Not pass action
                    from_territory, to_territory = attack_pairs[action]
                    if game_state.can_attack(1, from_territory, to_territory):
                        # Attack with all but 1 troop
                        num_troops = game_state.get_troops(from_territory) - 1
                        result = game_state.attack(1, from_territory, to_territory, num_troops)
                        
                        # Calculate reward
                        reward = 0
                        if result["territory_conquered"]:
                            reward = 5.0  # Reward for conquering territory
                            if len(game_state.get_owned_territories(1)) > len(game_state.get_owned_territories(2)):
                                reward += 2.0  # Bonus for having more territories
                        else:
                            # Reward based on damage dealt
                            reward = result["defender_losses"] * 0.5 - result["attacker_losses"] * 0.3
                    else:
                        reward = -0.1  # Penalty for invalid attack
                else:
                    reward = -0.2  # Penalty for passing
                
                # Add turn penalty
                reward -= 0.02 * turns
                
                # Get next state
                next_state = ppo_agent.get_state_vector(game_state)
                
                # Store experience
                ppo_agent.remember(current_state, action, action_prob, reward, next_state, done)
                total_reward += reward
                
                # Switch turns
                game_state.next_turn()
                turns += 1
                
                # Safety check for long games
                if turns > 250:
                    print(f"Game exceeded 250 turns, ending episode {episode + 1}")
                    total_reward -= 10.0
                    done = True
                    break
                
                # Print state every 10 turns for first 5 episodes
                if episode < 5 and turns % 10 == 0:
                    print(f"\nTurn {turns}:")
                    print(game_state)
            
            # Opponent bot's turn
            else:
                # Auto deploy and fortify for opponent
                auto_deploy(game_state, 2)
                auto_fortify(game_state, 2)
                
                # Get opponent's action
                from_territory, to_territory, num_troops = opponent_bot.get_action(game_state)
                
                # Execute opponent's action
                if from_territory is not None and to_territory is not None:
                    if game_state.can_attack(2, from_territory, to_territory):
                        game_state.attack(2, from_territory, to_territory, num_troops)
                
                # Switch turns
                game_state.next_turn()
            
            # Check if game is over
            if game_state.is_game_over():
                winner = game_state.get_winner()
                if winner == 1:
                    total_reward += 20.0  # Large reward for winning
                done = True
        
        # Update PPO agent
        if episode % update_every == 0:
            policy_loss, value_loss, entropy = ppo_agent.update()
            policy_losses.append(policy_loss)
            value_losses.append(value_loss)
            entropies.append(entropy)
        
        # Record metrics
        win_history.append(1 if game_state.get_winner() == 1 else 0)
        reward_history.append(total_reward)
        turn_history.append(turns)
        territory_counts.append(len(game_state.get_owned_territories(1)))
        
        # Print progress
        if (episode + 1) % 10 == 0:
            recent_win_rate = np.mean(win_history[-100:])
            recent_reward = np.mean(reward_history[-100:])
            recent_turns = np.mean(turn_history[-100:])
            recent_territories = np.mean(territory_counts[-100:])
            
            print(f"Episode {episode + 1} | Win Rate: {recent_win_rate:.2f} | Reward: {recent_reward:.2f} | Turns: {recent_turns:.1f} | "
                  f"PL: {policy_losses[-1]:.4f} | "
                  f"VL: {value_losses[-1]:.4f} | "
                  f"E: {entropies[-1]:.4f}")

    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Win rate
    plt.subplot(2, 3, 1)
    plt.plot(win_history)
    plt.title('Win Rate Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    
    # Territory control
    plt.subplot(2, 3, 2)
    plt.plot(territory_counts)
    plt.title('Territories Controlled')
    plt.xlabel('Episode')
    plt.ylabel('Number of Territories')
    
    # Rewards
    plt.subplot(2, 3, 3)
    plt.plot(reward_history)
    plt.title('Total Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    # Losses
    plt.subplot(2, 3, 4)
    plt.plot(policy_losses, label='Policy Loss')
    plt.plot(value_losses, label='Value Loss')
    plt.title('Losses Over Time')
    plt.xlabel('Update Step')
    plt.ylabel('Loss')
    plt.legend()
    
    # Entropy
    plt.subplot(2, 3, 5)
    plt.plot(entropies)
    plt.title('Policy Entropy')
    plt.xlabel('Update Step')
    plt.ylabel('Entropy')
    
    # Turn count
    plt.subplot(2, 3, 6)
    plt.plot(turn_history)
    plt.title('Turns per Game')
    plt.xlabel('Episode')
    plt.ylabel('Number of Turns')
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.close()

if __name__ == "__main__":
    # Example usage with RandomBot
    from simple.random_bot import RandomBot
    random_bot = RandomBot()
    train_ppo_against_bot(random_bot) 