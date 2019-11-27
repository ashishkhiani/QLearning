import random

import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.losses import mean_squared_error
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

    def get_action(self, state):
        if random.random() <= self.epsilon:
            # choose action via exploration
            return self.action_space.sample()

        # TODO choose action via exploitation
        return np.argmax(self.model.predict(state.reshape(1, 128)))

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
        states, actions, rewards, next_states, _ = list(map(np.array, list(zip(*batch))))

        current_q_values = self.model.predict(states)
        next_q_values = self.model.predict(next_states)
        target_q_values = rewards.reshape(len(batch), 1) + (next_q_values * self.discount_factor)
        loss = self.model.train_on_batch(states, target_q_values)
        print(loss)

        # states, actions, rewards, next_states, dones = list(map(np.array, list(zip(*batch))))
        #
        # targets = np.zeros((len(batch), self.action_space.n))
        #
        # for i in range(len(batch)):
        #     targets[i] = self.model.predict(states[i].reshape(1, 128), batch_size=1)
        #     fut_action = self.model.predict(next_states[i].reshape(1, 128), batch_size=1)
        #     targets[i, actions[i]] = rewards[i]
        #
        #     if not dones[i]:
        #         targets[i, actions[i]] += self.discount_factor * np.max(fut_action)
        #
        # loss = self.model.train_on_batch(states, targets)
        # print(loss)


    def save_network(self, path):
        # Saves model at specified path as h5 file
        self.model.save(path)
        print("Successfully saved network.")

    def load_network(self, path):
        self.model.load_weights(path)
        print("Succesfully loaded network.")
