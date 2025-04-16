import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import random
import torch.nn as nn
import torch


class PPONetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPONetwork, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, state):
        return self.actor(state), self.critic(state)

class PPOAgent:
    def __init__(self, player_id, state_dim, action_dim, lr=0.0003, gamma=0.99, epsilon=0.2):
        self.player_id = player_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PPONetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Memory for training
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        
        # Current episode memory
        self.current_log_probs = []
        self.current_values = []
        
    def act(self, state):
        """Alias for get_action that returns just the action"""
        action, _, _ = self.get_action(state)
        return action
        
    def get_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        action_probs, value = self.policy(state)
        dist = Categorical(torch.softmax(action_probs, dim=-1))
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # Store current step's log_prob and value
        self.current_log_probs.append(log_prob)
        self.current_values.append(value)
        
        return action.item(), log_prob, value
        
    def remember(self, state, action, reward):
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
        if len(self.states) == 0:  # Skip update if no experiences
            return
            
        # Convert lists to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.stack(self.log_probs).detach().to(self.device)
        old_values = torch.stack(self.values).detach().to(self.device)
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
        
        # PPO update
        for _ in range(10):  # Multiple epochs
            # Get new action probabilities and values
            action_probs, values = self.policy(states)
            dist = Categorical(torch.softmax(action_probs, dim=-1))
            new_log_probs = dist.log_prob(actions)
            
            # Calculate ratios and surrogate loss
            ratios = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = nn.MSELoss()(values.squeeze(), returns)
            
            # Total loss
            loss = actor_loss + 0.5 * value_loss
            
            # Update policy
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            
    def choose_deployments(self, game_state):
        state = game_state.get_state_vector()
        action, _, _ = self.get_action(state)
        
        # Convert action to deployment choices
        owned = game_state.get_owned_territories(self.player_id)
        num_troops = game_state.remaining_troops[self.player_id]
        
        # For now, deploy all troops to one territory
        # This is a simplification - we can make it more sophisticated later
        if num_troops > 0 and owned:
            chosen_territory = owned[action % len(owned)]
            return [chosen_territory] * num_troops
        return []
        
    def choose_attack(self, game_state):
        state = game_state.get_state_vector()
        action, _, _ = self.get_action(state)
        
        # Get all possible legal attacks
        legal_attacks = []
        for from_t in game_state.get_owned_territories(self.player_id):
            for to_t in game_state.adjacency.get(from_t, []):
                if game_state.can_attack(self.player_id, from_t, to_t):
                    for n in range(1, min(3, game_state.get_troops(from_t) - 1) + 1):
                        legal_attacks.append((from_t, to_t, n))
                        
        if legal_attacks:
            # Choose an attack based on the action
            return legal_attacks[action % len(legal_attacks)]
        return None
        
    def choose_fortify(self, game_state):
        return None  # skip for now
