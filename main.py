import gym

from DQN import DQN
from ReplayBuffer import ReplayBuffer
from parameters import EMULATION, NUM_EPISODES, REPLAY_BUFFER_CAPACITY, NUM_TIME_STEPS, REPLAY_BUFFER_SAMPLING_SIZE


def train_model_using_dqn(show_emulation=False):
    replay_buffer = ReplayBuffer(capacity=REPLAY_BUFFER_CAPACITY)
    env = gym.make(EMULATION)
    dqn = DQN(env.observation_space, env.action_space)

    for i in range(NUM_EPISODES):
        current_state = env.reset()
        for _ in range(NUM_TIME_STEPS):
            if show_emulation:
                env.render()

            # Select an action via explore or exploit
            action = dqn.get_action()

            # Execute selected action
            next_state, reward, done, info = env.step(action)

            # TODO handle done case

            # Store experience in replay buffer
            experience = (current_state, action, reward, next_state)
            replay_buffer.add(experience)

            # Sample batch from replay buffer
            if replay_buffer.can_sample(REPLAY_BUFFER_SAMPLING_SIZE):
                batch = replay_buffer.sample(REPLAY_BUFFER_SAMPLING_SIZE)

            # TODO Train Network with random sample

            # TODO Calculate loss between output Q-values and target Q-values

            # TODO do something with gradient descent and loss






    env.close()


if __name__ == "__main__":
    train_model_using_dqn(show_emulation=False)
