import random
from collections import deque
import numpy as np
import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim


# ---------- Q Network ----------
class QNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)



class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = map(np.array, zip(*batch))
        return s, a, r, s2, d

    def __len__(self):
        return len(self.buf)


def epsilon_by_frame(frame_idx, eps_start=1.0, eps_end=0.05, eps_decay=500_000):
    return max(0.05, 1.0 - frame_idx / 50_000)


@torch.no_grad()
def select_action(q_net, state, epsilon, action_dim, device):
    if random.random() < epsilon:
        return random.randrange(action_dim)
    s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    q = q_net(s)
    return int(torch.argmax(q, dim=1).item())


def train_dqn(
    env_id="LunarLander-v2",
    num_episodes=2000,
    buffer_size=200_000,
    batch_size=128,
    gamma=0.99,
    lr=3e-4,
    learn_start=10_000,
    train_freq=4,
    target_update_freq=500,
    max_steps_per_episode=1000,
    render=False,
    seed=0,
):
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print("device:", device)

    env = gym.make(env_id, render_mode="human" if render else None)
    env.reset(seed=seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # state = [
    #   x,            # 0:
    #   y,            # 1:
    #   vx,           # 2:
    #   vy,           # 3:
    #   angle,        # 4:
    #   angular_vel,  # 5:
    #   left_leg,     # 6:  if left leg on ground
    #   right_leg     # 7:
    # ]
    state_dim = env.observation_space.shape[0]
    # 4 action: do nothing, fire left engine, fire mid, fire right
    action_dim = env.action_space.n


    q_net = QNet(state_dim, action_dim).to(device)
    target_net = QNet(state_dim, action_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    replay = ReplayBuffer(buffer_size)

    total_steps = 0
    ep_rewards = []

    for ep in range(num_episodes):
        state, _ = env.reset()
        ep_reward = 0.0

        for t in range(max_steps_per_episode):
            eps = epsilon_by_frame(total_steps)

            action = select_action(q_net, state, eps, action_dim, device)
            next_state, reward, terminated, truncated, _ = env.step(action)
            reward = np.clip(reward, -5.0, 5.0)

            done = terminated or truncated

            replay.push(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
            total_steps += 1

            # there are enough experiences and every train_freq train
            if len(replay) >= learn_start and total_steps % train_freq == 0:
                s, a, r, s2, d = replay.sample(batch_size)

                s  = torch.tensor(s, dtype=torch.float32, device=device)
                a  = torch.tensor(a, dtype=torch.int64, device=device).unsqueeze(1)
                r  = torch.tensor(r, dtype=torch.float32, device=device).unsqueeze(1)
                s2 = torch.tensor(s2, dtype=torch.float32, device=device)
                d  = torch.tensor(d, dtype=torch.float32, device=device).unsqueeze(1)

                # Q(s,a)
                q_sa = q_net(s).gather(1, a)

                # target = r + gamma * max_a' Q_target(s',a')
                with torch.no_grad():
                    max_q_s2 = target_net(s2).max(dim=1, keepdim=True)[0]
                    target = r + gamma * (1.0 - d) * max_q_s2

                loss = nn.functional.mse_loss(q_sa, target)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
                optimizer.step()


            if total_steps % target_update_freq == 0 and len(replay) >= learn_start:
                target_net.load_state_dict(q_net.state_dict())

            if done:
                break

        ep_rewards.append(ep_reward)

        if (ep + 1) % 20 == 0:
            avg20 = np.mean(ep_rewards[-20:])
            print(f"Episode {ep+1:4d} | avg_reward(last20)={avg20:8.2f} | last={ep_reward:8.2f} | eps={eps:5.3f}")


        if len(ep_rewards) >= 100 and np.mean(ep_rewards[-100:]) >= 200:
            print(f"Solved! Episode {ep+1}, avg100={np.mean(ep_rewards[-100:]):.2f}")
            torch.save(q_net.state_dict(), "dqn_lunarlander.pth")
            break

    env.close()
    return q_net, ep_rewards

@torch.no_grad()
def play_dqn(
    model_path="dqn_lunarlander.pth",
    env_id="LunarLander-v2",
    episodes=5,
    seed=0
):
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print("device:", device)

    env = gym.make(env_id, render_mode="human")
    env.reset(seed=seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n


    q_net = QNet(state_dim, action_dim).to(device)
    q_net.load_state_dict(torch.load(model_path, map_location=device))
    q_net.eval()

    for ep in range(episodes):
        state, _ = env.reset()
        ep_reward = 0.0

        while True:
            s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            q_values = q_net(s)
            action = torch.argmax(q_values, dim=1).item()

            state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward

            if terminated or truncated:
                print(f"Episode {ep+1} reward: {ep_reward:.2f}")
                break

    env.close()

if __name__ == "__main__":
    qnet, rewards = train_dqn(render=False)
