import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import random
import torch.nn as nn
import torch


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim + 4, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.actor(state)

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_dim + 4, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.critic(state)

class PPOAgent:
    def __init__(self, player_id, state_dim, action_dim, 
                 lr=0.0003,
                 value_lr=0.001,
                 gamma=0.99,
                 epsilon=0.2,
                 entropy_coef=0.05):
        self.player_id = player_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Separate policy and value networks for each phase
        self.deployment_net = PolicyNetwork(state_dim, 10).to(self.device)  # 10 territories to deploy to
        self.attack_net = PolicyNetwork(state_dim, 30).to(self.device)     # 10 territories * 3 possible attack sizes
        self.fortify_net = PolicyNetwork(state_dim, 10).to(self.device)    # 10 territories to fortify to
        self.value_net = ValueNetwork(state_dim).to(self.device)
        
        # Separate optimizers
        self.deployment_optimizer = optim.Adam(self.deployment_net.parameters(), lr=lr)
        self.attack_optimizer = optim.Adam(self.attack_net.parameters(), lr=lr)
        self.fortify_optimizer = optim.Adam(self.fortify_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=value_lr)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.entropy_coef = entropy_coef
        self.state_dim = state_dim
        
        # Memory for training
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.phases = []  # Track which phase each action was taken in
        
        # Current episode memory
        self.current_log_probs = []
        self.current_values = []
        self.current_phases = []
        
        # Training stats
        self.training_step = 0
        
    def get_action(self, state, phase):
        # Get the appropriate network based on phase
        if phase == "deployment":
            net = self.deployment_net
        elif phase == "attack":
            net = self.attack_net
        else:  # fortify
            net = self.fortify_net
            
        state = torch.FloatTensor(state).to(self.device)
        
        # Get action probabilities from policy network
        action_logits = net(state)
        action_probs = torch.softmax(action_logits, dim=-1)
        
        # Get value from value network
        value = self.value_net(state)
        
        # Sample action
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # Store current step's log_prob, value, and phase
        self.current_log_probs.append(log_prob)
        self.current_values.append(value)
        self.current_phases.append(phase)
        
        return action.item(), log_prob, value
        
    def remember(self, state, action, reward, phase):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.phases.append(phase)
        
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
        self.current_phases = []
        
    def update(self):
        if len(self.states) == 0:
            return 0.0, 0.0, 0.0
            
        try:
            states = torch.FloatTensor(np.array(self.states)).to(self.device)
            actions = torch.LongTensor(self.actions).to(self.device)
            old_log_probs = torch.stack(self.log_probs).detach()
            old_values = torch.stack(self.values).detach()
            
            # Scale rewards to have mean 0 and std 1
            rewards = torch.FloatTensor(self.rewards).to(self.device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            
            # Calculate returns and advantages
            returns = []
            R = 0
            for r in reversed(rewards):
                R = r + self.gamma * R
                returns.insert(0, R)
            returns = torch.FloatTensor(returns).to(self.device)
            
            # Update value network multiple times to stabilize predictions
            for _ in range(3):
                value_pred = self.value_net(states).squeeze()
                value_loss = nn.MSELoss()(value_pred, returns)
                
                self.value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
                self.value_optimizer.step()
            
            # Recalculate value predictions for advantage estimation
            with torch.no_grad():
                value_pred = self.value_net(states).squeeze()
            
            # Calculate advantages
            advantages = returns - value_pred
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Update policy network multiple times
            total_policy_loss = 0
            total_entropy = 0
            
            for _ in range(3):  # Multiple policy updates per batch
                action_logits = self.deployment_net(states)
                action_probs = torch.softmax(action_logits, dim=-1)
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
                
                # Compute policy loss
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
                
                # Update policy network
                self.deployment_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.deployment_net.parameters(), 0.5)
                self.deployment_optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_entropy += entropy.item()
            
            self.training_step += 1
            
            return total_policy_loss / 3, value_loss.item(), total_entropy / 3
            
        except Exception as e:
            print(f"Error during update: {e}")
            print(f"States shape: {np.array(self.states).shape}")
            print(f"First state dim: {len(self.states[0])}")
            print(f"Last state dim: {len(self.states[-1])}")
            return 0.0, 0.0, 0.0
        
    def choose_deployments(self, game_state):
        state = game_state.get_state_vector()
        owned = game_state.get_owned_territories(self.player_id)
        num_troops = game_state.remaining_troops[self.player_id]
        
        if num_troops > 0 and owned:
            action, log_prob, value = self.get_action(state, "deployment")
            # Convert action to territory index
            territory = action % 10
            if territory in owned:
                return [territory] * num_troops
            else:
                # Fallback to random valid choice
                return [random.choice(owned)] * num_troops
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
            action, log_prob, _ = self.get_action(state, "attack")
            # Convert action to attack parameters
            attack_idx = action % len(legal_attacks)
            return legal_attacks[attack_idx]
        
        return None
        
    def choose_fortify(self, game_state):
        state = game_state.get_state_vector()
        owned = game_state.get_owned_territories(self.player_id)
        
        if len(owned) < 2:
            return None
            
        action, log_prob, _ = self.get_action(state, "fortify")
        # Convert action to territory index
        territory = action % 10
        if territory in owned:
            # Find a neighbor to fortify to
            for neighbor in game_state.adjacency.get(territory, []):
                if neighbor in owned:
                    return (territory, neighbor, game_state.get_troops(territory) - 1)
        
        return None
