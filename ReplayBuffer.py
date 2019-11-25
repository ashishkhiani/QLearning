import random
from collections import deque


class ReplayBuffer:

    def __init__(self, capacity):
        self.replay_buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def sample(self, n):
        """
        Returns a random sample of n elements from the replay buffer
        :param n:
        :return:
        """
        return random.sample(self.replay_buffer, n)

    def add(self, experience):
        """
        Adds an experience to the replay buffer
        :param experience: A tuple of the form (s_t, a_t, reward_next, s_next)
        """
        self.replay_buffer.append(experience)
