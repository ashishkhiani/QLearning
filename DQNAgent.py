import random

import numpy as np
from keras import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam

from ReplayBuffer import ReplayBuffer
from parameters import REPLAY_BUFFER_CAPACITY, REPLAY_BUFFER_SAMPLING_SIZE, LEARNING_RATE


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
        self.discount_factor = 0.95
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_shape=self.observation_space.shape))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_space.n, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))
        return model

    def get_action(self):
        if random.random() <= self.epsilon:
            # choose action via exploration
            return self.action_space.sample()

        # TODO choose action via exploitation
        return self.action_space.sample()

    def replay(self):

        if not self.replay_buffer.can_sample(REPLAY_BUFFER_SAMPLING_SIZE):
            return []

        batch = self.replay_buffer.sample(REPLAY_BUFFER_SAMPLING_SIZE)

        return batch

    def remember(self, experience):
        self.replay_buffer.add(experience)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay_rate)

    def learn(self, batch):
        if len(batch) == 0:
            return

        s_batch, a_batch, r_batch, s2_batch, d_batch = list(map(np.array, list(zip(*batch))))

        target = r_batch

        if not d_batch.all():
            target = r_batch + self.discount_factor * np.amax(self.model.predict(s2_batch))

        target_f = self.model.predict(s_batch)

        self.model.fit(s_batch, target_f, epochs=1, verbose=0)

    def save_network(self, path):
        # Saves model at specified path as h5 file
        self.model.save(path)
        print("Successfully saved network.")

    def load_network(self, path):
        self.model.load_weights(path)
        print("Succesfully loaded network.")
