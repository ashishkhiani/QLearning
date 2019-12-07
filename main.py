import os
import time

import gym
import pickle

from app.DQNAgent import DQNAgent
from app.DoubleDQNAgent import DoubleDQNAgent
from parameters import EMULATION, NUM_EPOCHS, FRAME_SKIP


def save_results(data, agent):
    """
    Saves the results to file
    :param data: Data to persist
    :param agent: Agent Class for file naming
    """
    print('Saving results')
    loss_values, reward_values, epsilon_values, time_values = data
    current_time = time.strftime('%Y_%m_%d_%H:%M:%S')
    directory = f'output/{current_time}'

    if agent:
        misc = f'{EMULATION}_{agent.__class__.__name__}'
    else:
        misc = f'{EMULATION}_Random'

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(f'{directory}/loss_values_{misc}.txt', 'wb') as file:
        pickle.dump(loss_values, file)

    with open(f'{directory}/reward_values_{misc}.txt', 'wb') as file:
        pickle.dump(reward_values, file)

    with open(f'{directory}/epsilon_values_{misc}.txt', 'wb') as file:
        pickle.dump(epsilon_values, file)

    with open(f'{directory}/time_values_{misc}.txt', 'wb') as file:
        pickle.dump(time_values, file)

    if agent:
        agent.save_network(f'{directory}/most_recent_model_{misc}.h5')


def baseline():
    """
    Baseline method that lets the agent perform only random actions
    """
    env = gym.make(EMULATION)
    reward_values = []
    time_values = []

    for i in range(NUM_EPOCHS):
        print(f'Epoch {i}')
        env.reset()
        time_steps = 0
        total_reward = 0
        done = False

        while not done:
            time_steps += 1
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)

            total_reward += reward

            if done:
                print(f'Epoch {i} ran for {time_steps} time steps')

        reward_values.append(total_reward)
        time_values.append(time_steps)
        print(f'TOTAL REWARD: {total_reward}')

    env.close()

    data = (None, reward_values, None, time_values)
    save_results(data, None)


def train_model(rl_agent, show_emulation=False, persist_data=False, initialize_buffer=True, normalize=True):
    """
    Trains the model using a certain Q-Learning Algorithm
    """

    # make gym environment
    env = gym.make(EMULATION)

    # initialize agent
    agent = rl_agent(env.observation_space, env.action_space)

    if initialize_buffer:
        # we populate buffer with random actions until maximum capacity
        agent.populate_buffer(env)

    loss_values = []
    reward_values = []
    epsilon_values = [agent.epsilon]
    time_values = []

    for i in range(NUM_EPOCHS):
        total_reward = 0
        print(f'Epoch {i}')
        current_state = env.reset()  # initialize current state
        done = False
        time_steps = 0

        while not done:  # epoch is over when agent loses all of its lives
            time_steps += 1

            if show_emulation:
                env.render()

            if normalize:  # normalize RAM state if required
                current_state = current_state.astype(float)
                normalize_state(current_state)

            # Select an action via explore or exploit
            action = agent.get_action(current_state)

            # Execute selected action
            next_state, reward, done, info = env.step(action)

            if normalize:
                next_state = next_state.astype(float)
                normalize_state(next_state)

            if FRAME_SKIP > 0:
                for _ in range(FRAME_SKIP - 1):  # repeat action for k frames
                    next_state, reward, done, info = env.step(action)

            if done:
                print(f'Epoch {i} ran for {time_steps} time steps')

            # Store experience in replay buffer
            experience = (current_state, action, reward, next_state, done)
            agent.remember(experience)

            # Update the current state
            current_state = next_state.copy()

            # Randomly sample from replay buffer
            batch = agent.replay()

            # Train network with random sample
            if len(batch) > 0:
                loss = agent.learn(batch)
                loss_values.append(loss)

            total_reward += reward

        # decay epsilon at the end of every epoch
        agent.decay_epsilon()
        print(f'TOTAL REWARD: {total_reward}')

        reward_values.append(total_reward)
        epsilon_values.append(agent.epsilon)
        time_values.append(time_steps)

    env.close()

    if persist_data:
        # save results
        data = (loss_values, reward_values, epsilon_values, time_values)
        save_results(data, agent)


def play_game(path, num_episodes, use_random=False):
    """
    Updates the neural network with the saved model on file and runs the emulation
    """
    env = gym.make(EMULATION)
    agent = DQNAgent(env.observation_space, env.action_space)
    agent.load_network(path)

    for i_episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            env.render()
            if use_random:
                action = env.action_space.sample()
            else:
                action = agent.predict_action(state)

            state, reward, done, info = env.step(action)
            if done:
                break
    env.close()


def normalize_state(state):
    """
    Reduce the values of the RAM state to be in the range 0 <= j <= 1
    """
    for j in range(len(state)):
        state[j] = state[j] / 256


def main():
    """
    Main code to run program
    """
    train_model(
        rl_agent=DoubleDQNAgent,
        persist_data=True,
        initialize_buffer=True,
        show_emulation=False,
        normalize=False
    )

    train_model(
        rl_agent=DQNAgent,
        persist_data=True,
        initialize_buffer=True,
        show_emulation=False,
        normalize=False
    )

    baseline()
