import time

import gym
import pickle

from app.DQNAgent import DQNAgent
from app.DoubleDQNAgent import DoubleDQNAgent
from parameters import EMULATION, NUM_EPOCHS, FRAME_SKIP


def save_results(loss_values, reward_values, epsilon_values, time_values):
    print('Saving results')
    current_time = time.time()
    with open(f'output/loss_values_{EMULATION}_{current_time}.txt', 'wb') as file:
        pickle.dump(loss_values, file)

    with open(f'output/reward_values_{EMULATION}_{current_time}.txt', 'wb') as file:
        pickle.dump(reward_values, file)

    with open(f'output/epsilon_values_{EMULATION}_{current_time}.txt', 'wb') as file:
        pickle.dump(epsilon_values, file)

    with open(f'output/time_values_{EMULATION}_{current_time}.txt', 'wb') as file:
        pickle.dump(time_values, file)


def train_model(rl_agent, show_emulation=False, persist_data=False, initialize_buffer=True, normalize=True):
    env = gym.make(EMULATION)
    agent = rl_agent(env.observation_space, env.action_space)

    if initialize_buffer:
        agent.populate_buffer(env)

    loss_values = []
    reward_values = []
    epsilon_values = [agent.epsilon]
    time_values = []

    max_reward = 0

    for i in range(NUM_EPOCHS):
        total_reward = 0
        print(f'Epoch {i}')
        current_state = env.reset()
        done = False
        time_stamps = 0

        while not done:
            time_stamps += 1

            if show_emulation:
                env.render()

            if normalize:
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
                for _ in range(FRAME_SKIP - 1):
                    next_state, reward, done, info = env.step(action)

            if done:
                print(f'Epoch {i} ran for {time_stamps} timestamps')

            # Store experience in replay buffer
            experience = (current_state, action, reward, next_state, done)
            agent.remember(experience)

            current_state = next_state

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
        time_values.append(time_stamps)

        # save best model
        if total_reward >= max_reward:
            max_reward = total_reward
            agent.save_network('output/models/best_model.h5')

    env.close()

    if persist_data:
        save_results(loss_values, reward_values, epsilon_values, time_values)

    agent.save_network('output/models/most_recent_model.h5')


def play_game(model_name, num_episodes, use_random=False):
    env = gym.make(EMULATION)
    agent = DQNAgent(env.observation_space, env.action_space)
    agent.load_network('static/' + model_name)

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
    for j in range(len(state)):
        state[j] = state[j] / 256


if __name__ == "__main__":
    train_model(
        rl_agent=DoubleDQNAgent,
        persist_data=True,
        initialize_buffer=False,
        show_emulation=False,
        normalize=True
    )
