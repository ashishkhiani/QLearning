import gym

from DQNAgent import DQNAgent
from parameters import EMULATION, NUM_EPOCHS


def train_model_using_dqn(show_emulation=False):
    env = gym.make(EMULATION)
    agent = DQNAgent(env.observation_space, env.action_space)

    for i in range(NUM_EPOCHS):
        total_reward = 0
        print(f'Epoch {i}')
        current_state = env.reset()
        done = False
        j = 0
        while not done:
            if show_emulation:
                env.render()

            # Select an action via explore or exploit
            action = agent.get_action(current_state)

            # Execute selected action
            next_state, reward, done, info = env.step(action)

            if done:
                print(f'Epoch {i} ended at the {j} timestamp')

            # Store experience in replay buffer
            experience = (current_state, action, reward, next_state, done)
            agent.remember(experience)

            current_state = next_state

            batch = agent.replay()

            # Train network with random sample
            if len(batch) > 0:
                agent.learn(batch)

            total_reward += reward
            j += 1

        # decay epsilon at the end of every epoch
        agent.decay_epsilon()
        print(f'TOTAL REWARD: {total_reward}')

    env.close()


if __name__ == "__main__":
    train_model_using_dqn(show_emulation=False)
