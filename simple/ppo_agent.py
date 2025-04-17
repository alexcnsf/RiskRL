import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        # Input layer: 31-dimensional state (10 troops + 10 owners + 1 turn flag + 10 adjacency)
        self.fc1 = nn.Linear(state_dim, 64)
        self.ln1 = nn.LayerNorm(64)
        
        # Hidden layer
        self.fc2 = nn.Linear(64, 64)
        self.ln2 = nn.LayerNorm(64)
        
        # Output layer: action_dim possible actions
        self.fc3 = nn.Linear(64, action_dim)
        
        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = self.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        return torch.softmax(self.fc3(x), dim=-1)

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        # Input layer: 31-dimensional state (10 troops + 10 owners + 1 turn flag + 10 adjacency)
        self.fc1 = nn.Linear(state_dim, 64)
        self.ln1 = nn.LayerNorm(64)
        
        # Hidden layer
        self.fc2 = nn.Linear(64, 64)
        self.ln2 = nn.LayerNorm(64)
        
        # Output layer: single value
        self.fc3 = nn.Linear(64, 1)
        
        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = self.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        return self.fc3(x)

class SimplePPOAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99, epsilon=0.2, entropy_coef=0.01):
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.value_net = ValueNetwork(state_dim)
        self.optimizer = optim.Adam(list(self.policy_net.parameters()) + list(self.value_net.parameters()), lr=learning_rate)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.entropy_coef = entropy_coef
        
        self.memory = []
        
    def get_action(self, game_state):
        """Get action from policy network"""
        state = game_state.get_state_vector()
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.policy_net(state_tensor)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), action_probs[0][action.item()].item()
        
    def remember(self, state, action, action_prob, reward, next_state, done):
        self.memory.append((state, action, action_prob, reward, next_state, done))
        
    def update(self):
        if len(self.memory) == 0:
            return 0, 0, 0  # Return zero losses if no memory
            
        # Convert memory to tensors
        states = torch.FloatTensor(np.array([m[0] for m in self.memory]))
        actions = torch.LongTensor([m[1] for m in self.memory])
        old_probs = torch.FloatTensor([m[2] for m in self.memory])
        rewards = torch.FloatTensor([m[3] for m in self.memory])
        next_states = torch.FloatTensor(np.array([m[4] for m in self.memory]))
        dones = torch.FloatTensor([m[5] for m in self.memory])
        
        # Calculate returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Get value predictions
        values = self.value_net(states).squeeze()
        
        # Calculate advantages
        advantages = returns - values.detach()
        
        # Get new action probabilities
        action_probs = self.policy_net(states)
        dist = Categorical(action_probs)
        new_probs = dist.log_prob(actions)
        
        # Calculate ratio
        ratio = torch.exp(new_probs - old_probs)
        
        # Calculate surrogate loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Calculate value loss
        value_loss = F.mse_loss(values, returns)
        
        # Calculate entropy
        entropy = dist.entropy().mean()
        
        # Total loss
        loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Clear memory
        self.memory = []
        
        return policy_loss.item(), value_loss.item(), entropy.item() 