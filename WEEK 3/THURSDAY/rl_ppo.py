import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass

# PPO with a shared trunk (actor + critic), GAE advantages, and clipped objective.# Tuned for clarity and CPU friendliness.

@dataclass
class PPOConfig:
    env_name: str = "CartPole-v1"
    gamma: float = 0.99
    lam: float = 0.95          # GAE lambda
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    lr: float = 3e-3
    epochs: int = 6            # policy epochs per batch
    batch_episodes: int = 8    # how many episodes per update
    max_updates: int = 120
    hidden: int = 128

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
        )
        self.pi = nn.Sequential(
            nn.Linear(hidden, action_dim),
            nn.Softmax(dim=-1)
        )
        self.v = nn.Linear(hidden, 1)

    def forward(self, s):
        z = self.trunk(s)
        probs = self.pi(z)
        value = self.v(z).squeeze(-1)
        return probs, value

    def get_action(self, s_np):
        s = torch.tensor(s_np, dtype=torch.float32)
        probs, value = self.forward(s)
        dist = torch.distributions.Categorical(probs)
        a = dist.sample()
        logp = dist.log_prob(a)
        ent = dist.entropy()
        return a.item(), logp.detach(), value.detach(), ent.detach()

def rollout_batch(env, model, cfg: PPOConfig):
    traj = {
        "obs": [], "acts": [], "rews": [], "dones": [],
        "logps": [], "vals": [], "ents": [], "ep_rews": []
    }
    for _ in range(cfg.batch_episodes):
        s, _ = env.reset()
        ep_rew, done = 0.0, False
        while not done:
            a, logp, v, ent = model.get_action(s)
            s2, r, term, trunc, _ = env.step(a)
            done = term or trunc

            traj["obs"].append(s)
            traj["acts"].append(a)
            traj["rews"].append(r)
            traj["dones"].append(done)
            traj["logps"].append(logp.item())
            traj["vals"].append(v.item())
            traj["ents"].append(ent.item())

            ep_rew += r
            s = s2
        traj["ep_rews"].append(ep_rew)

    # Bootstrap value for final state as 0 (episodic env)
    traj["vals"].append(0.0)
    return traj

def compute_gae(traj, cfg: PPOConfig):
    rews = np.array(traj["rews"], dtype=np.float32)
    vals = np.array(traj["vals"], dtype=np.float32)  # has extra bootstrap at end
    dones = np.array(traj["dones"], dtype=np.bool_)
    T = len(rews)

    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        delta = rews[t] + cfg.gamma * vals[t+1] * (1.0 - float(dones[t])) - vals[t]
        gae = delta + cfg.gamma * cfg.lam * (1.0 - float(dones[t])) * gae
        adv[t] = gae
    ret = adv + vals[:-1]
    # Normalize advantages
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return adv, ret

def ppo_update(model, optimizer, obs, acts, old_logps, adv, ret, cfg: PPOConfig):
    for _ in range(cfg.epochs):
        probs, values = model(obs)
        dist = torch.distributions.Categorical(probs)
        logps = dist.log_prob(acts)
        entropy = dist.entropy().mean()

        ratio = torch.exp(logps - old_logps)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv
        policy_loss = -torch.mean(torch.min(surr1, surr2))

        value_loss = nn.functional.mse_loss(values, ret)
        loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def train_ppo(cfg=PPOConfig()):
    env = gym.make(cfg.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = ActorCritic(state_dim, action_dim, hidden=cfg.hidden)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    best = 0.0
    for upd in range(1, cfg.max_updates + 1):
        traj = rollout_batch(env, model, cfg)
        adv, ret = compute_gae(traj, cfg)

        obs = torch.tensor(np.array(traj["obs"], dtype=np.float32))
        acts = torch.tensor(np.array(traj["acts"], dtype=np.int64))
        old_logps = torch.tensor(np.array(traj["logps"], dtype=np.float32))
        adv = torch.tensor(adv, dtype=torch.float32)
        ret = torch.tensor(ret, dtype=torch.float32)

        ppo_update(model, optimizer, obs, acts, old_logps, adv, ret, cfg)

        avg_rew = float(np.mean(traj["ep_rews"]))
        best = max(best, avg_rew)
        print(f"[PPO] Update {upd:03d} | AvgEpRew {avg_rew:.1f} | BestAvg {best:.1f} | Episodes {cfg.batch_episodes}")

        if avg_rew >= 500.0:
            print("[PPO] Solved! 🎉")
            break

    env.close()
    return model

if __name__ == "__main__":
    _ = train_ppo()
