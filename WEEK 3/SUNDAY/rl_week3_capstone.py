import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass

# ========= Config & Model =========
@dataclass
class CFG:
    env_name: str = "CartPole-v1"
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    lr: float = 3e-3
    epochs: int = 6
    batch_episodes: int = 8
    max_updates: int = 120
    hidden: int = 128
    early_stop_window: int = 10
    early_stop_threshold: float = 475.0  # average return over window
    max_eval_episodes: int = 5

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU())
        self.pi = nn.Sequential(nn.Linear(hidden, action_dim), nn.Softmax(dim=-1))
        self.v  = nn.Linear(hidden, 1)

    def forward(self, s):
        z = self.trunk(s)
        probs = self.pi(z)
        value = self.v(z).squeeze(-1)
        return probs, value

    @torch.no_grad()
    def act(self, state_np):
        s = torch.tensor(state_np, dtype=torch.float32)
        probs, _ = self.forward(s)
        dist = torch.distributions.Categorical(probs)
        a = dist.sample().item()
        return a, probs.numpy()

# ========= Rollout, GAE, Update =========
def rollout_batch(env, model, cfg: CFG):
    buf = {"obs": [], "acts": [], "rews": [], "dones": [], "logps": [], "vals": [], "ep_rews": []}
    for _ in range(cfg.batch_episodes):
        s, _ = env.reset()
        done, ep_rew = False, 0.0
        while not done:
            s_t = torch.tensor(s, dtype=torch.float32)
            probs, v = model(s_t)
            dist = torch.distributions.Categorical(probs)
            a = dist.sample()
            s2, r, term, trunc, _ = env.step(a.item())
            done = term or trunc
            buf["obs"].append(s)
            buf["acts"].append(a.item())
            buf["rews"].append(r)
            buf["dones"].append(float(done))
            buf["logps"].append(dist.log_prob(a).item())
            buf["vals"].append(v.item())
            ep_rew += r
            s = s2
        buf["ep_rews"].append(ep_rew)
    buf["vals"].append(0.0)  # episodic bootstrap
    return buf

def compute_gae(buf, cfg: CFG):
    rews  = np.array(buf["rews"], dtype=np.float32)
    vals  = np.array(buf["vals"], dtype=np.float32)
    dones = np.array(buf["dones"], dtype=np.float32)
    T = len(rews)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        delta = rews[t] + cfg.gamma * vals[t+1] * (1 - dones[t]) - vals[t]
        gae = delta + cfg.gamma * cfg.lam * (1 - dones[t]) * gae
        adv[t] = gae
    ret = adv + vals[:-1]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    obs   = torch.tensor(np.array(buf["obs"], dtype=np.float32))
    acts  = torch.tensor(np.array(buf["acts"], dtype=np.int64))
    oldlp = torch.tensor(np.array(buf["logps"], dtype=np.float32))
    adv_t = torch.tensor(adv, dtype=torch.float32)
    ret_t = torch.tensor(ret, dtype=torch.float32)
    return obs, acts, oldlp, adv_t, ret_t

def ppo_update(model, opt, obs, acts, oldlp, adv, ret, cfg: CFG):
    policy_losses, value_losses, entropies = [], [], []
    for _ in range(cfg.epochs):
        probs, values = model(obs)
        dist = torch.distributions.Categorical(probs)
        logps = dist.log_prob(acts)
        ratio = torch.exp(logps - oldlp)

        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv
        policy_loss = -torch.mean(torch.min(surr1, surr2))
        value_loss  = nn.functional.mse_loss(values, ret)
        entropy     = dist.entropy().mean()

        loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
        opt.zero_grad()
        loss.backward()
        opt.step()

        policy_losses.append(policy_loss.item())
        value_losses.append(value_loss.item())
        entropies.append(entropy.item())
    return np.mean(policy_losses), np.mean(value_losses), np.mean(entropies)

# ========= Safety Layer & Evaluation =========
class SafetyLayer:
    def __init__(self, x_soft=2.2, theta_soft=0.21):
        self.x_soft = x_soft
        self.theta_soft = theta_soft
    def allow(self, s):
        x, x_dot, th, th_dot = s
        return (abs(x) <= self.x_soft) and (abs(th) <= self.theta_soft)

def evaluate_with_safety(model, cfg: CFG, episodes=5):
    env = gym.make(cfg.env_name)
    safety = SafetyLayer()
    returns = []
    for _ in range(episodes):
        s, _ = env.reset()
        done, ep_rew = False, 0.0
        while not done:
            if not safety.allow(s):
                a = 0  # safe fallback
            else:
                a, _ = model.act(s)
            s, r, term, trunc, _ = env.step(a)
            done = term or trunc
            ep_rew += r
        returns.append(ep_rew)
    env.close()
    return float(np.mean(returns))

# ========= Training Loop with Diagnostics & Early Stop =========
def train_capstone(cfg=CFG()):
    env = gym.make(cfg.env_name)
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n
    model = ActorCritic(s_dim, a_dim, hidden=cfg.hidden)
    opt = optim.Adam(model.parameters(), lr=cfg.lr)

    reward_window = []
    for upd in range(1, cfg.max_updates + 1):
        buf = rollout_batch(env, model, cfg)
        obs, acts, oldlp, adv, ret = compute_gae(buf, cfg)
        pol_loss, val_loss, ent = ppo_update(model, opt, obs, acts, oldlp, adv, ret, cfg)

        avg_ret = float(np.mean(buf["ep_rews"]))
        reward_window.append(avg_ret)
        if len(reward_window) > cfg.early_stop_window:
            reward_window.pop(0)
        window_avg = float(np.mean(reward_window))

        print(f"[PPO] Upd {upd:03d} | AvgRet {avg_ret:.1f} | WindowAvg {window_avg:.1f} | "
              f"PolicyLoss {pol_loss:.3f} | ValueLoss {val_loss:.3f} | Ent {ent:.3f}")

        # Early stopping criteria: sustained high performance
        if len(reward_window) == cfg.early_stop_window and window_avg >= cfg.early_stop_threshold:
            print("[Early Stop] Stable high returns achieved.")
            break

    env.close()
    # Evaluate with safety wrapper
    eval_ret = evaluate_with_safety(model, cfg, episodes=cfg.max_eval_episodes)
    print(f"[Eval+Safety] Mean return over {cfg.max_eval_episodes} episodes: {eval_ret:.1f}")
    return model

if __name__ == "__main__":
    _ = train_capstone()
