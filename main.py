import gym

from DQN import DQN
from parameters import EMULATION, NUM_EPISODES


def train_model_using_dqn(show_emulation=False):
    env = gym.make(EMULATION)
    dqn = DQN(env.observation_space, env.action_space)
    env.reset()

    for _ in range(NUM_EPISODES):
        if show_emulation:
            env.render()

        action = dqn.get_action()
        env.step(action)

    env.close()


if __name__ == "__main__":
    train_model_using_dqn(show_emulation=True)
