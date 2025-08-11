import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict

# -------------------------------# Part A: Policy Gradient (REINFORCE)# -------------------------------
class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.net(x)

def select_action_policy(policy, state):
    s = torch.tensor(state, dtype=torch.float32)
    probs = policy(s)
    dist = torch.distributions.Categorical(probs)
    a = dist.sample()
    logp = dist.log_prob(a)
    return a.item(), logp

def compute_returns(rewards, gamma=0.99):
    G, out = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        out.append(G)
    out.reverse()
    out = torch.tensor(out, dtype=torch.float32)
    out = (out - out.mean()) / (out.std() + 1e-8)
    return out

def train_reinforce(env_name="CartPole-v1", episodes=200, hidden=128, lr=1e-2, gamma=0.99):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]  
    action_dim = env.action_space.n
    policy = PolicyNet(state_dim, hidden, action_dim)
    opt = optim.Adam(policy.parameters(), lr=lr)

    best = 0.0
    rewards_hist = []

    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        done = False
        logps, rewards = [], []
        ep_rew = 0.0

        while not done:
            action, logp = select_action_policy(policy, state)
            nxt, r, term, trunc, _ = env.step(action)
            done = term or trunc
            logps.append(logp)
            rewards.append(r)
            ep_rew += r
            state = nxt

        returns = compute_returns(rewards, gamma)
        logps = torch.stack(logps)
        loss = -(logps * returns).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

        best = max(best, ep_rew)
        rewards_hist.append(ep_rew)
        print(f"[REINFORCE] Ep {ep:03d} | Reward {ep_rew:.1f} | Best {best:.1f}")

        if ep_rew >= 500:
            print("[REINFORCE] Solved!")
            break

    env.close()
    return rewards_hist

# -------------------------------# Part B: Simple Value-Based Baseline (Q-learning on Discretized States)# Warning: For CartPole, raw states are continuous, so we discretize approx.# -------------------------------
def discretize_state(obs, bins):
    # obs: [x, x_dot, theta, theta_dot]
    upper = np.array([2.4, 3.0, 0.2095, 3.5])
    lower = -upper
    ratios = (obs - lower) / (upper - lower)
    ratios = np.clip(ratios, 0, 1)
    return tuple(int(r * (b - 1)) for r, b in zip(ratios, bins))

def train_q_learning(env_name="CartPole-v1", episodes=500, gamma=0.99, alpha=0.1, eps=1.0, eps_min=0.05, eps_decay=0.995):
    env = gym.make(env_name)
    action_n = env.action_space.n

    bins = (9, 9, 9, 9)
    Q = defaultdict(lambda: np.zeros(action_n))
    best = 0.0
    rewards_hist = []

    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        s_disc = discretize_state(np.array(state), bins)
        done, ep_rew = False, 0.0

        while not done:
            if np.random.rand() < eps:
                a = env.action_space.sample()
            else:
                a = int(np.argmax(Q[s_disc]))

            nxt, r, term, trunc, _ = env.step(a)
            done = term or trunc
            s_disc_next = discretize_state(np.array(nxt), bins)

            td_target = r + (0 if done else gamma * np.max(Q[s_disc_next]))
            td_error = td_target - Q[s_disc][a]
            Q[s_disc][a] += alpha * td_error

            ep_rew += r
            s_disc = s_disc_next

        eps = max(eps_min, eps * eps_decay)
        best = max(best, ep_rew)
        rewards_hist.append(ep_rew)
        print(f"[Q-Learning] Ep {ep:03d} | Reward {ep_rew:.1f} | Best {best:.1f} | eps {eps:.3f}")

        if ep_rew >= 500:
            print("[Q-Learning] Solved!")
            break

    env.close()
    return rewards_hist

if __name__ == "__main__":
    r1 = train_reinforce(episodes=220)
    r2 = train_q_learning(episodes=600)
