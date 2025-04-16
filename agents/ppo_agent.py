import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import random
import torch.nn as nn
import torch


class PPONetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPONetwork, self).__init__()
        
        # Increased input dimension to handle additional features
        self.actor = nn.Sequential(
            nn.Linear(state_dim + 4, 128),  # +4 for max additional features
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, action_dim)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim + 4, 128),  # +4 for max additional features
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
    def forward(self, state):
        # Handle single state case for action selection
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.actor(state), self.critic(state)

class PPOAgent:
    def __init__(self, player_id, state_dim, action_dim, 
                 lr=0.0001,  # Reduced learning rate
                 gamma=0.95,  # Reduced gamma for faster reward propagation
                 epsilon=0.1,  # Reduced clip range
                 entropy_coef=0.01):  # Added entropy coefficient
        self.player_id = player_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PPONetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.entropy_coef = entropy_coef
        self.state_dim = state_dim  # Store state_dim for padding
        
        # Memory for training
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        
        # Current episode memory
        self.current_log_probs = []
        self.current_values = []
        
        # Training stats
        self.training_step = 0
        
    def act(self, state):
        """Alias for get_action that returns just the action"""
        action, _, _ = self.get_action(state)
        return action
        
    def get_action(self, state):
        # Pad state if needed
        if len(state) < self.state_dim + 4:
            padding = np.zeros(self.state_dim + 4 - len(state))
            state = np.concatenate([state, padding])
            
        state = torch.FloatTensor(state).to(self.device)
        action_probs, value = self.policy(state)
        
        # Add exploration noise in early training
        if self.training_step < 1000:
            action_probs = action_probs + torch.randn_like(action_probs) * 0.1
            
        dist = Categorical(torch.softmax(action_probs, dim=-1))
        
        # Add entropy bonus for exploration
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # Store current step's log_prob and value
        self.current_log_probs.append(log_prob)
        self.current_values.append(value)
        
        return action.item(), log_prob, value
        
    def remember(self, state, action, reward):
        # Ensure state has consistent dimension by padding if needed
        if len(state) < self.state_dim + 4:
            padding = np.zeros(self.state_dim + 4 - len(state))
            state = np.concatenate([state, padding])
            
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        
        # Add the stored log_prob and value for this step
        if self.current_log_probs:
            self.log_probs.append(self.current_log_probs.pop(0))
        if self.current_values:
            self.values.append(self.current_values.pop(0))
        
    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.current_log_probs = []
        self.current_values = []
        
    def update(self):
        if len(self.states) == 0:
            return
            
        # Convert lists to tensors, ensuring consistent dimensions
        try:
            states = torch.FloatTensor(np.array(self.states)).to(self.device)
            actions = torch.LongTensor(self.actions).to(self.device)
            old_log_probs = torch.stack(self.log_probs).detach()
            old_values = torch.stack(self.values).detach()
            rewards = torch.FloatTensor(self.rewards).to(self.device)
            
            # Calculate returns and advantages
            returns = []
            advantages = []
            R = 0
            for r in reversed(rewards):
                R = r + self.gamma * R
                returns.insert(0, R)
            returns = torch.FloatTensor(returns).to(self.device)
            advantages = returns - old_values.squeeze()
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # PPO update with more epochs and early stopping
            best_loss = float('inf')
            no_improve_count = 0
            
            for epoch in range(20):  # Increased epochs
                # Get new action probabilities and values
                action_probs, values = self.policy(states)
                dist = Categorical(torch.softmax(action_probs, dim=-1))
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
                
                # Calculate ratios and surrogate loss
                ratios = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages
                
                # Actor and critic losses
                actor_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.MSELoss()(values.squeeze(), returns)
                
                # Total loss with entropy bonus
                loss = actor_loss + 0.5 * value_loss - self.entropy_coef * entropy
                
                # Early stopping check
                if loss < best_loss:
                    best_loss = loss
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count >= 3:  # Stop if no improvement for 3 epochs
                        break
                
                # Update policy
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
        
        except Exception as e:
            print(f"Error during update: {e}")
            print(f"States shape: {np.array(self.states).shape}")
            print(f"First state dim: {len(self.states[0])}")
            print(f"Last state dim: {len(self.states[-1])}")
        
        self.training_step += 1
        
        # Decay exploration noise and entropy coefficient
        if self.training_step % 100 == 0:
            self.entropy_coef = max(0.001, self.entropy_coef * 0.995)
        
    def choose_deployments(self, game_state):
        state = game_state.get_state_vector()
        owned = game_state.get_owned_territories(self.player_id)
        num_troops = game_state.remaining_troops[self.player_id]
        
        if num_troops > 0 and owned:
            action_probs = []
            for territory in range(10):
                # Add territory-specific features (now matching network input size)
                territory_features = [
                    float(territory in owned),  # Ownership indicator
                    float(num_troops) / 10.0,   # Available troops (normalized)
                    0.0,  # Padding
                    0.0   # Padding
                ]
                
                state_with_territory = np.concatenate([state, territory_features])
                action, log_prob, value = self.get_action(state_with_territory)
                # Convert log_prob to scalar probability
                prob = float(np.exp(log_prob.cpu().detach().numpy())) if territory in owned else 0.0
                action_probs.append(prob)
            
            # Normalize probabilities
            action_probs = np.array(action_probs, dtype=np.float32)
            if action_probs.sum() > 0:
                action_probs = action_probs / action_probs.sum()
                # Choose territory based on learned probabilities
                chosen_territory = np.random.choice(range(10), p=action_probs)
                return [chosen_territory] * num_troops
            else:
                # Fallback to random valid choice if all probs are zero
                chosen_territory = np.random.choice(list(owned))
                return [chosen_territory] * num_troops
        return []
        
    def choose_attack(self, game_state):
        state = game_state.get_state_vector()
        legal_attacks = []
        
        # Get all possible legal attacks
        for from_t in game_state.get_owned_territories(self.player_id):
            for to_t in game_state.adjacency.get(from_t, []):
                if game_state.can_attack(self.player_id, from_t, to_t):
                    for n in range(1, min(3, game_state.get_troops(from_t) - 1) + 1):
                        legal_attacks.append((from_t, to_t, n))
                        
        if legal_attacks:
            attack_scores = []
            for from_t, to_t, n in legal_attacks:
                # Create feature vector for this attack (now matching network input size)
                attack_features = [
                    game_state.get_troops(from_t) / 10.0,  # Normalize troop counts
                    game_state.get_troops(to_t) / 10.0,
                    n / 3.0,  # Normalize number of attacking troops
                    len([t for t in game_state.adjacency.get(to_t, []) 
                         if game_state.get_owner(t) == self.player_id]) / len(game_state.adjacency.get(to_t, []))
                ]
                
                state_with_attack = np.concatenate([state, attack_features])
                action, log_prob, _ = self.get_action(state_with_attack)
                # Convert log_prob to scalar probability
                prob = float(np.exp(log_prob.cpu().detach().numpy()))
                attack_scores.append(prob)
            
            # Convert scores to probabilities
            attack_scores = np.array(attack_scores, dtype=np.float32)
            if attack_scores.sum() > 0:
                attack_probs = attack_scores / attack_scores.sum()
                # Choose attack based on learned probabilities
                chosen_idx = np.random.choice(len(legal_attacks), p=attack_probs)
                return legal_attacks[chosen_idx]
            else:
                # Fallback to random valid choice if all probs are zero
                return random.choice(legal_attacks)
        
        return None
        
    def choose_fortify(self, game_state):
        return None  # skip for now
