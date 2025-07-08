import gymnasium as gym

env = gym.make('LunarLander-v3')
state_shape = env.observation_space.shape
state_size = env.observation_space.shape[0]
number_of_actions = env.action_space.n

print('state shape: ', state_shape)
print('state size: ', state_size)
print('number of actions: ', number_of_actions)