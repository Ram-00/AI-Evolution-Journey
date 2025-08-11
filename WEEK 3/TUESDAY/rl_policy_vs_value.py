import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict

# ===== Policy Gradient (REINFORCE) Implementation =====
class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.network(x)

def select_action(policy, state):
    state_tensor = torch.tensor(state, dtype=torch.float32)
    probs = policy(state_tensor)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action.item(), log_prob

def compute_returns(rewards, gamma=0.99):
    G = 0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns

def train_reinforce(env, episodes=200, gamma=0.99):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy = PolicyNet(state_dim, 128, action_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    
    rewards_history = []
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        log_probs = []
        rewards = []
        total_reward = 0
        
        while not done:
            action, log_prob = select_action(policy, state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            log_probs.append(log_prob)
            rewards.append(reward)
            total_reward += reward
            state = next_state
            
        returns = compute_returns(rewards, gamma)
        loss = -torch.sum(torch.stack(log_probs) * returns)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        rewards_history.append(total_reward)
        print(f"[REINFORCE] Episode {episode+1}: {total_reward}")
        
        if total_reward >= 500:
            print("[REINFORCE] Solved!")
            break
    
    return rewards_history

# ===== Value-Based (Q-learning) Implementation =====
def discretize_state(state, bins=(10, 10, 10, 10)):
    if isinstance(state, tuple):
        state = np.array(state)
    upper = np.array([2.4, 3.0, 0.2095, 3.5])
    lower = -upper
    ratios = (state - lower) / (upper - lower)
    ratios = np.clip(ratios, 0, 1)
    discretized = tuple((ratios * (np.array(bins) - 1)).astype(int))
    return discretized

def train_q_learning(env, episodes=600, gamma=0.99, alpha=0.1, 
                    epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    rewards_history = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        discretized_state = discretize_state(state)
        done = False
        total_reward = 0
        
        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(Q[discretized_state]))
                
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            next_discretized_state = discretize_state(next_state)
            old_value = Q[discretized_state][action]
            next_max = np.max(Q[next_discretized_state])
            target = reward + gamma * next_max * (1 - done)
            Q[discretized_state][action] += alpha * (target - old_value)
            
            discretized_state = next_discretized_state
            total_reward += reward
            
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        rewards_history.append(total_reward)
        print(f"[Q-Learning] Episode {episode+1}: {total_reward}")
        
        if total_reward >= 500:
            print("[Q-Learning] Solved!")
            break
            
    return rewards_history

def main():
    env = gym.make('CartPole-v1')
    
    print("Training REINFORCE (Policy Gradient)...")
    reinforce_rewards = train_reinforce(env)
    
    print("\nTraining Q-Learning (Value-Based)...")
    q_learning_rewards = train_q_learning(env)
    
    print(f"\nREINFORCE best: {max(reinforce_rewards)}")
    print(f"Q-Learning best: {max(q_learning_rewards)}")

if __name__ == '__main__':
    main()
