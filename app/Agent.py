import random
from abc import abstractmethod

import numpy as np
from keras import Sequential
from keras.engine.saving import load_model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

from app.ReplayBuffer import ReplayBuffer

from parameters import \
    REPLAY_BUFFER_CAPACITY, \
    EPSILON_DECAY_RATE, \
    EPSILON_MIN, \
    DISCOUNT_FACTOR, \
    LEARNING_RATE, \
    REPLAY_BUFFER_SAMPLING_SIZE


class Agent:

    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        self.replay_buffer = ReplayBuffer(capacity=REPLAY_BUFFER_CAPACITY)
        self.epsilon = 1
        self.epsilon_decay_rate = EPSILON_DECAY_RATE
        self.epsilon_min = EPSILON_MIN
        self.discount_factor = DISCOUNT_FACTOR
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(128, input_shape=self.observation_space.shape))

        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.35))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.35))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.35))

        model.add(Dense(self.action_space.n, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))
        return model

    def predict_action(self, state):
        new_shape = (1,) + self.observation_space.shape
        return np.argmax(self.model.predict(state.reshape(new_shape)))

    def get_action(self, state):
        if random.random() <= self.epsilon:
            # choose action via exploration
            return self.action_space.sample()

        #  choose action via exploitation
        return self.predict_action(state)

    def replay(self):

        if not self.replay_buffer.can_sample(REPLAY_BUFFER_SAMPLING_SIZE):
            return []

        batch = self.replay_buffer.sample(REPLAY_BUFFER_SAMPLING_SIZE)

        return batch

    def remember(self, experience):
        self.replay_buffer.add(experience)

    @abstractmethod
    def learn(self, batch):
        pass

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay_rate)

    def save_network(self, path):
        self.model.save(path)
        print("Successfully saved network.")

    def load_network(self, path):
        self.model = load_model(path)
        print("Succesfully loaded network.")