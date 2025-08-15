import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict

# ---------------------------# Utility: Reward variants# ---------------------------
def shaped_reward(obs, reward, done):
    # obs = [x, x_dot, theta, theta_dot]
    x, x_dot, th, th_dot = obs
    # Encourage small |x| and |theta| (keep near center and upright)
    center_bonus = max(0.0, 1.0 - min(1.0, abs(x) / 2.4))
    upright_bonus = max(0.0, 1.0 - min(1.0, abs(th) / 0.2095))
    bonus = 0.05 * center_bonus + 0.1 * upright_bonus
    # Preserve terminal penalty behavior via env default reward (CartPole gives +1 each step)
    return reward + bonus

def sparse_reward(_obs, reward, done):
    # Use the environment’s default reward (step = +1); no shaping
    return reward

# ---------------------------# PPO-style Actor-Critic agent (entropy exploration)# ---------------------------
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
        )
        self.pi = nn.Sequential(nn.Linear(hidden, action_dim), nn.Softmax(dim=-1))
        self.v = nn.Linear(hidden, 1)

    def forward(self, s):
        z = self.trunk(s)
        probs = self.pi(z)
        value = self.v(z).squeeze(-1)
        return probs, value

def run_batch_ppo(env, model, batch_episodes, use_shaping=True, gamma=0.99, lam=0.95):
    transitions = []
    ep_returns = []

    for _ in range(batch_episodes):
        s, _ = env.reset()
        done, ep_ret = False, 0.0
        while not done:
            s_t = torch.tensor(s, dtype=torch.float32)
            probs, v = model(s_t)
            dist = torch.distributions.Categorical(probs)
            a = dist.sample()
            s2, r, term, trunc, _ = env.step(a.item())
            done = term or trunc
            # Reward variant
            r_adj = shaped_reward(s2, r, done) if use_shaping else sparse_reward(s2, r, done)

            transitions.append((s, a.item(), r_adj, float(done), v.item(), dist.log_prob(a).item()))
            ep_ret += r
            s = s2
        ep_returns.append(ep_ret)

    # Add bootstrap value 0 because episodes end (episodic)
    return transitions, np.array(ep_returns, dtype=np.float32)

def compute_gae(transitions, gamma=0.99, lam=0.95, state_dim=4):
    # Unroll arrays correctly based on the 6-tuple: (s, a, r, done, v, logp)
    states = np.array([t[0] for t in transitions], dtype=np.float32)
    acts   = np.array([t[1] for t in transitions], dtype=np.int64)
    rews   = np.array([t[2] for t in transitions], dtype=np.float32)
    dones  = np.array([t[3] for t in transitions], dtype=np.float32)
    vals   = np.array([t[4] for t in transitions], dtype=np.float32)
    logps  = np.array([t[5] for t in transitions], dtype=np.float32)

    # Add terminal bootstrap value 0 because episodes end (episodic)
    vals_ext = np.concatenate([vals, [0.0]])
    T = len(rews)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        delta = rews[t] + gamma * vals_ext[t+1] * (1 - dones[t]) - vals[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        adv[t] = gae
    ret = adv + vals
    # Normalize
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    return states, acts, logps, adv, ret

def ppo_update(model, optimizer, states, acts, old_logps, adv, ret,
               clip_eps=0.2, value_coef=0.5, entropy_coef=0.01, epochs=6):
    states_t = torch.tensor(states, dtype=torch.float32)
    acts_t   = torch.tensor(acts, dtype=torch.int64)
    oldlp_t  = torch.tensor(old_logps, dtype=torch.float32)
    adv_t    = torch.tensor(adv, dtype=torch.float32)
    ret_t    = torch.tensor(ret, dtype=torch.float32)

    for _ in range(epochs):
        probs, values = model(states_t)
        dist = torch.distributions.Categorical(probs)
        logps = dist.log_prob(acts_t)
        ratio = torch.exp(logps - oldlp_t)
        surr1 = ratio * adv_t
        surr2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * adv_t
        policy_loss = -torch.mean(torch.min(surr1, surr2))
        value_loss = nn.functional.mse_loss(values, ret_t)
        entropy = dist.entropy().mean()
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def train_ppo_variant(env_name="CartPole-v1", episodes_per_update=8, max_updates=80,
                      use_shaping=True, entropy_coef=0.01, lr=3e-3):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]  # Changed from env.observation_space.shape
    action_dim = env.action_space.n
    model = ActorCritic(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_avg = 0.0
    for upd in range(1, max_updates + 1):
        transitions, ep_returns = run_batch_ppo(env, model, batch_episodes=episodes_per_update,
                                                use_shaping=use_shaping)
        states, acts, old_logps, adv, ret = compute_gae(transitions)
        ppo_update(model, optimizer, states, acts, old_logps, adv, ret, entropy_coef=entropy_coef)

        avg = float(np.mean(ep_returns))
        best_avg = max(best_avg, avg)
        variant = "Shaped+Entropy" if use_shaping else "Sparse+Entropy"
        print(f"[PPO {variant}] Upd {upd:03d} | AvgReturn {avg:.1f} | BestAvg {best_avg:.1f}")

        if avg >= 500.0:
            print(f"[PPO {variant}] Solved!")
            break

    env.close()
    return best_avg

# ---------------------------# Q-learning baseline (epsilon-greedy)# ---------------------------
def discretize_state(obs, bins=(9,9,9,9)):
    upper = np.array([2.4, 3.0, 0.2095, 3.5])
    lower = -upper
    ratios = (obs - lower) / (upper - lower)
    ratios = np.clip(ratios, 0, 1)
    return tuple(int(r * (b - 1)) for r, b in zip(ratios, bins))

def train_qlearning(env_name="CartPole-v1", episodes=600, gamma=0.99, alpha=0.1,
                    eps=1.0, eps_min=0.05, eps_decay=0.995, use_shaping=True):
    env = gym.make(env_name)
    action_n = env.action_space.n
    Q = defaultdict(lambda: np.zeros(action_n))
    best = 0.0

    for ep in range(1, episodes + 1):
        s, _ = env.reset()
        s_disc = discretize_state(np.array(s))
        done, ep_ret = False, 0.0

        while not done:
            if np.random.rand() < eps:
                a = env.action_space.sample()
            else:
                a = int(np.argmax(Q[s_disc]))

            s2, r, term, trunc, _ = env.step(a)
            done = term or trunc
            r_adj = shaped_reward(s2, r, done) if use_shaping else sparse_reward(s2, r, done)

            s2_disc = discretize_state(np.array(s2))
            td_target = r_adj + (0 if done else gamma * np.max(Q[s2_disc]))
            td_error = td_target - Q[s_disc][a]
            Q[s_disc][a] += alpha * td_error

            ep_ret += r
            s_disc = s2_disc

        eps = max(eps_min, eps * eps_decay)
        best = max(best, ep_ret)
        variant = "Shaped+εgreedy" if use_shaping else "Sparse+εgreedy"
        print(f"[Q-Learning {variant}] Ep {ep:03d} | Return {ep_ret:.1f} | Best {best:.1f} | eps {eps:.3f}")

        if ep_ret >= 500:
            print(f"[Q-Learning {variant}] Solved!")
            break

    env.close()
    return best

if __name__ == "__main__":
    print("=== PPO with shaped reward + entropy ===")
    best_ppo_shaped = train_ppo_variant(use_shaping=True, entropy_coef=0.01)

    print("\n=== PPO with sparse reward + entropy ===")
    best_ppo_sparse = train_ppo_variant(use_shaping=False, entropy_coef=0.01)

    print("\n=== Q-learning with shaped reward + ε-greedy ===")
    best_q_shaped = train_qlearning(use_shaping=True)

    print("\n=== Q-learning with sparse reward + ε-greedy ===")
    best_q_sparse = train_qlearning(use_shaping=False)

    print("\nSummary:")
    print(f"PPO (Shaped): {best_ppo_shaped:.1f}")
    print(f"PPO (Sparse): {best_ppo_sparse:.1f}")
    print(f"Q  (Shaped): {best_q_shaped:.1f}")
    print(f"Q  (Sparse): {best_q_sparse:.1f}")
