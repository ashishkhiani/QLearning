import gym
import pickle

from DQNAgent import DQNAgent
from QLearningDataHandler import QLearningDataHandler
from parameters import EMULATION, NUM_EPOCHS


def plot(loss_values, reward_values, epsilon_values, time_values):
    with open('output/loss_values.txt', 'wb') as file:
        pickle.dump(loss_values, file)

    with open('output/reward_values.txt', 'wb') as file:
        pickle.dump(reward_values, file)

    with open('output/epsilon_values.txt', 'wb') as file:
        pickle.dump(epsilon_values, file)

    with open('output/time_values.txt', 'wb') as file:
        pickle.dump(time_values, file)

    handler = QLearningDataHandler()
    handler.plot_loss_curve(loss_values)
    handler.plot_reward_curve(reward_values)
    handler.plot_epsilon_curve(epsilon_values)
    handler.plot_time_taken_curve(time_values)


def train_model_using_dqn(show_emulation=False):
    env = gym.make(EMULATION)
    agent = DQNAgent(env.observation_space, env.action_space)

    loss_values = []
    reward_values = []
    epsilon_values = [agent.epsilon]
    time_values = []

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

            # Select an action via explore or exploit
            action = agent.get_action(current_state)

            # Execute selected action
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
                loss_values.append(agent.learn(batch))

            total_reward += reward

        # decay epsilon at the end of every epoch
        agent.decay_epsilon()
        print(f'TOTAL REWARD: {total_reward}')

        reward_values.append(total_reward)
        epsilon_values.append(agent.epsilon)
        time_values.append(time_stamps)

    env.close()

    plot(loss_values, reward_values, epsilon_values, time_values)


if __name__ == "__main__":
    train_model_using_dqn(show_emulation=False)
