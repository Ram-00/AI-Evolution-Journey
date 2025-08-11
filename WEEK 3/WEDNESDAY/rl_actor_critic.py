import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# A2C-style Actor-Critic for CartPole-v1# - Shared MLP trunk# - Actor head: categorical policy over actions# - Critic head: state-value V(s)# - Loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

class ActorCritic(nn.Module):
    def __init__(self, state_dim, hidden=128, action_dim=2):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        z = self.backbone(x)
        probs = self.actor(z)
        value = self.critic(z).squeeze(-1)
        return probs, value

def run_episode(env, model, gamma=0.99):
    s, _ = env.reset()
    done = False

    logps = []
    values = []
    rewards = []
    entropies = []
    ep_reward = 0.0

    while not done:
        s_t = torch.tensor(s, dtype=torch.float32)
        probs, value = model(s_t)
        dist = torch.distributions.Categorical(probs)
        a = dist.sample()
        logp = dist.log_prob(a)
        entropy = dist.entropy().mean()

        s2, r, term, trunc, _ = env.step(a.item())
        done = term or trunc

        logps.append(logp)
        values.append(value)
        rewards.append(r)
        entropies.append(entropy)
        ep_reward += r
        s = s2

    # Compute returns and advantages
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    returns_t = torch.tensor(returns, dtype=torch.float32)
    values_t = torch.stack(values)
    advantages = returns_t - values_t.detach()

    return logps, values_t, returns_t, advantages, torch.stack(entropies), ep_reward

def train_a2c(env_name="CartPole-v1",
              episodes=250,
              gamma=0.99,
              lr=3e-3,
              value_coef=0.5,
              entropy_coef=0.01):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = ActorCritic(state_dim, hidden=128, action_dim=action_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best = 0.0
    for ep in range(1, episodes + 1):
        logps, values, returns, advantages, entropies, ep_rew = run_episode(env, model, gamma)

        # Losses
        policy_loss = -(torch.stack(logps) * advantages).sum()
        value_loss = nn.functional.mse_loss(values, returns)
        entropy_bonus = entropies.mean()

        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        best = max(best, ep_rew)
        print(f"[A2C] Ep {ep:03d} | Reward {ep_rew:.1f} | Best {best:.1f} | "
              f"Loss {loss.item():.3f} | Vloss {value_loss.item():.3f} | Ent {entropy_bonus.item():.3f}")

        if ep_rew >= 500:
            print("[A2C] Solved! 🎉")
            break

    env.close()
    return model

if __name__ == "__main__":
    _ = train_a2c()
