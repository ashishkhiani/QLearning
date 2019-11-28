import numpy as np
from keras.engine.saving import load_model

from app.Agent import Agent
from parameters import TARGET_MODEL_UPDATE_ITERATIONS


class DoubleDQNAgent(Agent):
    """
    Double Deep Q-Networks
    """

    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)
        self.target_model = self.build_model()
        self.iterations = 0

    def learn(self, batch):
        self.iterations += 1

        if self.iterations == TARGET_MODEL_UPDATE_ITERATIONS:
            self.clone_model()
            self.iterations = 0

        states, actions, rewards, next_states, dones = list(map(np.array, list(zip(*batch))))

        next_q_values = self.target_model.predict(next_states)

        is_not_done = np.logical_not(dones.reshape(len(batch), 1))  # Flip all Ts to Fs and Fs to Ts.
        target_q_values = rewards.reshape(len(batch), 1) + (next_q_values * self.discount_factor * is_not_done)

        loss = self.model.train_on_batch(states, target_q_values)
        return loss

    def clone_model(self, persist=False):
        if persist:
            self.save_network('output\\models\\clone.h5')
            self.target_model = self.load_network('output\\models\\clone.h5')
        else:
            model_weights = self.model.get_weights()
            self.target_model.set_weights(model_weights)

    def load_target_network(self, path):
        self.target_model = load_model(path)
        print("Succesfully target loaded network.")
