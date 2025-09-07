from collections import deque
import numpy as np
from Agent import Agent
from Constans import state_size, number_of_actions, env
import torch

agent = Agent(state_size, number_of_actions)
number_episodes = 2000
maximum_number_timesteps_per_episode = 1000
epsilon_starting_value = 1.0
epsilon_ending_value = 0.01
epsilon_decay_value = 0.995
epsilon = epsilon_starting_value
''' explain scores_on_100_episodes deque 
    deque is a double ended queue that stores the last 100 values of scores on 100 episodes
    maxlen=100 means that the deque will only store the last 100 values of scores on 100 episodes
'''
scores_on_100_episodes = deque(maxlen=100)

for episode in range(1,number_episodes+1):
    state, _ =env.reset()
    score = 0
    for t in range(maximum_number_timesteps_per_episode):
        action = agent.act(state, epsilon)
        next_state, reward, done, _ ,_= env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            break
    scores_on_100_episodes.append(score)
    epsilon = max(epsilon_ending_value, epsilon_decay_value * epsilon)
    print('\rEpisode {}\tAverage Score: {:.2f}\tEpsilon: {:.2f}'.format(episode, np.mean(scores_on_100_episodes), epsilon), end="")
    if episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\tEpsilon: {:.2f}'.format(episode, np.mean(scores_on_100_episodes), epsilon))
    if np.mean(scores_on_100_episodes) >= 200.0:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode-100, np.mean(scores_on_100_episodes)))
        torch.save(agent.local_qnetwork.state_dict(), 'checkpoint.pth')
        break

