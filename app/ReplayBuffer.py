import random
from collections import deque


class ReplayBuffer:

    def __init__(self, capacity):
        """
        Initialize replay buffer to have size equal to the capacity
        :param capacity: Max size of buffer
        """
        self.replay_buffer = deque(maxlen=capacity)

    def can_sample(self, n):
        return len(self.replay_buffer) >= n

    def sample(self, n):
        """
        :param n: Number of elements to sample
        :return: A random sample of n elements from the replay buffer
        """
        return random.sample(self.replay_buffer, n)

    def add(self, experience):
        """
        Adds an experience to the replay buffer
        :param experience: A tuple of the form (s_t, a_t, reward_next, s_next)
        """
        self.replay_buffer.append(experience)
