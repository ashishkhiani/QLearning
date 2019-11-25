import gym

from DQNAgent import DQNAgent
from parameters import EMULATION, NUM_EPISODES, NUM_TIME_STEPS


def train_model_using_dqn(show_emulation=False):
    env = gym.make(EMULATION)
    agent = DQNAgent(env.observation_space, env.action_space)

    for i in range(NUM_EPISODES):
        current_state = env.reset()
        for _ in range(NUM_TIME_STEPS):
            if show_emulation:
                env.render()

            # Select an action via explore or exploit
            action = agent.get_action()

            # Execute selected action
            next_state, reward, done, info = env.step(action)

            # TODO handle done case

            # Store experience in replay buffer
            experience = (current_state, action, reward, next_state)
            agent.remember(experience)

            next_state = current_state

            agent.replay()

            # TODO Train Network with random sample

            # TODO Calculate loss between output Q-values and target Q-values

            # TODO do something with gradient descent and loss

        # decay epsilon at the end of every episode
        agent.decay_epsilon()

    env.close()


if __name__ == "__main__":
    train_model_using_dqn(show_emulation=False)
