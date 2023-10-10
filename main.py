import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(episodes, is_training=True, render=False):
    env = gym.make('FrozenLake-v1', map_name='8x8', is_slippery=False, render_mode='human' if render else None)

    if(is_training):
        q = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        f = open("frozenLake8x8.pkl", "rb")
        q = pickle.load(f)
        f.close()

    learning_rate_a = 0.9 # alpha learning rate
    discount_factor_g = 0.9 # gamma or discount factor

    epsilon = 1                    # 1 = 100% random actions, 0 = 0% random actions
    epsilon_decay_rate = 0.0001    # epsilon decay rate. 1/0.0001 = 10,000 episodes to decay to 0
    rng = np.random.default_rng()  # random number generator

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0] # states: 0 to 63, 0=top-left, 63=bottom-right corner
        terminated = False     # True if the agent fell into a hole or reached the goal
        truncated = False      # True when the actions > 200

        while not terminated and not truncated:
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample() # 0=left, 1=down, 2=right, 3=up
            else:
                action = np.argmax(q[state, :])
            

            new_state, reward, terminated, truncated, _ = env.step(action)

            if is_training:
                q[state, action] = q[state, action] + learning_rate_a * (reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action])

            state = new_state
    
        epsilon = max(epsilon - epsilon_decay_rate, 0)

        if(epsilon == 0):
            learning_rate_a = 0.0001
        
        if reward == 1:
            rewards_per_episode[i] = 1

    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.savefig('frozenLake8x8.png')

    if is_training:
        f = open("frozenLake8x8.pkl", "wb")
        pickle.dump(q, f)
        f.close()

    print("State-action values: {}, Reward: {}".format(state, reward))

if __name__ == '__main__':
    run(17500)
    run(1, is_training=False, render=True)






















































# import gymnasium as gym

# def run():
#     env = gym.make('FrozenLake-v1', map_name='8x8', is_slippery=True, render_mode='human')

#     state = env.reset()[0]
#     terminated = False
#     truncated = False

#     while not terminated and not truncated:

#         action = env.action_space.sample()

#         new_state, reward, terminated, truncated, _ = env.step(action)

#         state = new_state
    
#     env.close()

# if __name__ == '__main__':
#     run()