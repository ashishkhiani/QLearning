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
    LEARNING_RATE, \
    REPLAY_BUFFER_SAMPLING_SIZE


class Agent:
    """
    Abstract class that manages the different Q-Learning Algorithms discussed
    """

    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        self.replay_buffer = ReplayBuffer(capacity=REPLAY_BUFFER_CAPACITY)
        self.epsilon = 1
        self.model = self.build_model()

    def build_model(self):
        """
        Create the neural network architecture
        """
        model = Sequential()
        # input layer
        model.add(Dense(128, input_shape=self.observation_space.shape))

        # hidden layers
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.35))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.35))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.35))

        # output layer
        model.add(Dense(self.action_space.n, activation=None))
        model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))

        return model

    def predict_action(self, state):
        """
        Using the neural network, select action with highest Q-Value
        """
        new_shape = (1,) + self.observation_space.shape
        return np.argmax(self.model.predict(state.reshape(new_shape)))

    def get_action(self, state):
        """
        Return action to be performed in current state
        """
        if random.random() <= self.epsilon:
            # choose action via exploration
            return self.action_space.sample()

        #  choose action via exploitation
        return self.predict_action(state)

    def replay(self):
        """
        Sample replay buffer for a mini-batch
        """
        if not self.replay_buffer.can_sample(REPLAY_BUFFER_SAMPLING_SIZE):
            return []

        batch = self.replay_buffer.sample(REPLAY_BUFFER_SAMPLING_SIZE)

        return batch

    def remember(self, experience):
        """
        Add experience to the replay buffer
        """
        self.replay_buffer.add(experience)

    def populate_buffer(self, env):
        """
        Initialize replay buffer with random actions until it reaches maximum capacity
        """
        print('Populating buffer')
        current_state = env.reset()
        for i in range(REPLAY_BUFFER_CAPACITY):
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            experience = (current_state, action, reward, next_state, done)
            self.remember(experience)
            current_state = next_state.copy()

            if done:
                current_state = env.reset()

    @abstractmethod
    def learn(self, batch):
        """
        Abstract method to be implemented by child class
        """
        pass

    def decay_epsilon(self):
        """
        Reduce the value of epsilon linearly by the decay rate
        """
        if self.epsilon > EPSILON_MIN:
            self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY_RATE)

    def save_network(self, path):
        """
        Save neural network to file
        """
        self.model.save(path)
        print("Successfully saved network.")

    def load_network(self, path):
        """
        Load neural network from file
        """
        self.model = load_model(path)
        print("Succesfully loaded network.")
