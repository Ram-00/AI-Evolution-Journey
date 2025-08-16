import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass

# ============= PPO policy (small, single-file) =============
@dataclass
class PPOCfg:
    env_name: str = "CartPole-v1"
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    lr: float = 3e-3
    epochs: int = 6
    batch_episodes: int = 8
    max_updates: int = 100
    hidden: int = 128

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU())
        self.pi = nn.Sequential(nn.Linear(hidden, action_dim), nn.Softmax(dim=-1))
        self.v  = nn.Linear(hidden, 1)

    def forward(self, s_t):
        z = self.trunk(s_t)
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

def rollout_batch(env, model, cfg: PPOCfg):
    traj = {"obs": [], "acts": [], "rews": [], "dones": [], "logps": [], "vals": [], "ep_rews": []}
    for _ in range(cfg.batch_episodes):
        s, _ = env.reset()
        ep_rew, done = 0.0, False
        while not done:
            s_t = torch.tensor(s, dtype=torch.float32)
            probs, v = model(s_t)
            dist = torch.distributions.Categorical(probs)
            a = dist.sample()
            s2, r, term, trunc, _ = env.step(a.item())
            done = term or trunc

            traj["obs"].append(s)
            traj["acts"].append(a.item())
            traj["rews"].append(r)
            traj["dones"].append(float(done))
            traj["logps"].append(dist.log_prob(a).item())
            traj["vals"].append(v.item())

            ep_rew += r
            s = s2
        traj["ep_rews"].append(ep_rew)
    traj["vals"].append(0.0)  # bootstrap 0 at episode ends (episodic)
    return traj

def compute_gae(traj, cfg: PPOCfg):
    rews  = np.array(traj["rews"], dtype=np.float32)
    vals  = np.array(traj["vals"], dtype=np.float32)  # includes bootstrap at end
    dones = np.array(traj["dones"], dtype=np.float32)
    T = len(rews)

    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        delta = rews[t] + cfg.gamma * vals[t+1] * (1 - dones[t]) - vals[t]
        gae = delta + cfg.gamma * cfg.lam * (1 - dones[t]) * gae
        adv[t] = gae
    ret = adv + vals[:-1]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    obs = torch.tensor(np.array(traj["obs"], dtype=np.float32))
    acts = torch.tensor(np.array(traj["acts"], dtype=np.int64))
    oldlp = torch.tensor(np.array(traj["logps"], dtype=np.float32))
    adv_t = torch.tensor(adv, dtype=torch.float32)
    ret_t = torch.tensor(ret, dtype=torch.float32)
    return obs, acts, oldlp, adv_t, ret_t

def ppo_update(model, optimizer, obs, acts, oldlp, adv, ret, cfg: PPOCfg):
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
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def train_ppo(cfg=PPOCfg()):
    env = gym.make(cfg.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = ActorCritic(state_dim, action_dim, hidden=cfg.hidden)
    opt = optim.Adam(model.parameters(), lr=cfg.lr)

    best = 0.0
    for upd in range(1, cfg.max_updates + 1):
        traj = rollout_batch(env, model, cfg)
        obs, acts, oldlp, adv, ret = compute_gae(traj, cfg)
        ppo_update(model, opt, obs, acts, oldlp, adv, ret, cfg)

        avg = float(np.mean(traj["ep_rews"]))
        best = max(best, avg)
        print(f"[PPO Train] Update {upd:03d} | AvgEpRew {avg:.1f} | BestAvg {best:.1f}")
        if avg >= 500.0:
            print("[PPO Train] Solved!")
            break
    env.close()
    return model

# ============= Safety/constraint layer and agent-style wrapper =============

class SafetyLayer:
    """
    Minimal example: refuse action if cart position or pole angle exceeds soft thresholds.
    Add your own domain-specific checks (budgets, rate limits, allowlists).
    """
    def __init__(self, x_limit=2.2, theta_limit=0.2):
        self.x_limit = x_limit
        self.theta_limit = theta_limit

    def check(self, state_np, action):
        x, x_dot, th, th_dot = state_np
        if abs(x) > self.x_limit or abs(th) > self.theta_limit:
            return False, "State out of safe bounds"
        return True, "OK"

class RLPolicyTool:
    """
    Exposes the trained PPO policy as a callable 'tool'.
    The agent can pass state, and this tool returns an action if safe, else a safe fallback.
    """
    def __init__(self, model: ActorCritic, safety: SafetyLayer, safe_fallback=0):
        self.model = model
        self.safety = safety
        self.safe_fallback = safe_fallback  # e.g., push-left as a conservative default

    def __call__(self, state_np):
        ok, msg = self.safety.check(state_np, None)
        if not ok:
            return {"action": self.safe_fallback, "reason": f"fallback due to safety: {msg}"}
        a, probs = self.model.act(state_np)
        return {"action": int(a), "probs": probs.tolist(), "reason": "policy action"}

def demo_agent_with_tool(model: ActorCritic):
    env = gym.make("CartPole-v1")
    safety = SafetyLayer(x_limit=2.2, theta_limit=0.21)
    tool = RLPolicyTool(model, safety, safe_fallback=0)

    s, _ = env.reset()
    done, ep_rew, steps = False, 0.0, 0
    while not done and steps < 300:
        # The 'agent' would normally run reasoning/RAG here; we just call the tool.
        out = tool(s)
        a = out["action"]
        s, r, term, trunc, _ = env.step(a)
        done = term or trunc
        ep_rew += r
        steps += 1
    env.close()
    print(f"[Safe Agent] Episode reward with safety/tool: {ep_rew:.1f}, steps={steps}")

if __name__ == "__main__":
    # 1) Train PPO locally (on-policy, simulated) — stand-in for offline pretraining + safe finetune.
    model = train_ppo()

    # 2) Wrap the trained policy as a safe tool and run a quick agent demo.
    demo_agent_with_tool(model)
