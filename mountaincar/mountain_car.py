import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(episodes, is_training=True, render=False):

    # render_mode='human' to enable graph visible, but make training slowly
    env = gym.make('MountainCar-v0', render_mode='human' if render else None)

    # Box([-1.2  -0.07], [0.6  0.07], (2,), float32)
    # print(env.observation_space)
    # Divide position and velocity into segments
    # split position and velocity, take 20 scale points evenly
    # pos_space，vel_space store the bin edges
    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)

    if(is_training):
        # env.action_space.n represent 3 action, to left, to right, stay still
        # q[i][j][z] represents at position i execute z with speed j
        q = np.zeros((len(pos_space), len(vel_space), env.action_space.n))
    else:
        f = open('mountain_car.pkl', 'rb')
        q = pickle.load(f)
        f.close()
    # A learning rate of 0.2 leads to more stable and smoother convergence,
    # while a learning rate of 0.9 causes faster but highly unstable learning with significant oscillations.
    learning_rate_a = 0.9
    discount_factor_g = 0.9

    # 100% random actions
    epsilon = 1
    epsilon_decay_rate = 2/episodes
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        # Starting position, starting velocity
        state = env.reset()[0]

        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)

        terminated = False

        rewards=0

        while(not terminated and rewards>-1000):

            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                # q[state_p, state_v, :]
                # [-7.22658067 -5.09462109 -5.7260531 ] represents the Q of each action,
                # so in this example, index 0 has the max Q, so take action 0 is best choice.
                action = np.argmax(q[state_p, state_v, :])

            new_state,reward,terminated,_,_ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)

            if is_training:
                # r+γ max Q(s′,a) − Q(s,a)
                q[state_p, state_v, action] = q[state_p, state_v, action] + learning_rate_a * (
                    reward + discount_factor_g*np.max(q[new_state_p, new_state_v,:]) - q[state_p, state_v, action]
                )

            state_p = new_state_p
            state_v = new_state_v

            rewards+=reward


        print("Episode: {}, Reward: {}".format(i, rewards))

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        rewards_per_episode[i] = rewards

    env.close()

    # Save Q table to file
    if is_training:
        f = open('mountain_car.pkl', 'wb')
        pickle.dump(q, f)
        f.close()

    mean_rewards = np.zeros(episodes)
    for t in range(episodes):
        mean_rewards[t] = np.mean(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(mean_rewards)
    plt.savefig(f'mountain_car.png')


    rewards = rewards_per_episode
    window = 200

    # 计算滑动平均
    moving_avg = np.convolve(
        rewards,
        np.ones(window) / window,
        mode='valid'
    )

    plt.figure(figsize=(10, 5))


    plt.plot(rewards, color='steelblue', alpha=0.3, label='reward per episode')


    plt.plot(
        range(window - 1, len(rewards)),
        moving_avg,
        color='red',
        linewidth=2,
        label=f'average reward over the last {window} episodes'
    )

    plt.xlabel('Episode #')
    plt.ylabel('Total Reward')
    plt.title('Episode Reward History')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    run(1000, is_training=True, render=False)

    # run(1000, is_training=False, render=True)
