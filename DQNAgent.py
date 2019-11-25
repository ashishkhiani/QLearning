import random

from ReplayBuffer import ReplayBuffer
from parameters import REPLAY_BUFFER_CAPACITY, REPLAY_BUFFER_SAMPLING_SIZE


class DQNAgent:
    """
    Deep Q-Networks
    """

    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        self.replay_buffer = ReplayBuffer(capacity=REPLAY_BUFFER_CAPACITY)
        self.epsilon = 1
        self.epsilon_decay_rate = 0.995
        self.epsilon_min = 0.1

    def get_action(self):
        if random.random() <= self.epsilon:
            # choose action via exploration
            return self.action_space.sample()

        # TODO choose action via exploitation
        return self.action_space.sample()

    def replay(self):

        if not self.replay_buffer.can_sample(REPLAY_BUFFER_SAMPLING_SIZE):
            return

        batch = self.replay_buffer.sample(REPLAY_BUFFER_SAMPLING_SIZE)

        # TODO train network with batch

    def remember(self, experience):
        self.replay_buffer.add(experience)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay_rate)