import numpy as np

from app.Agent import Agent
from parameters import TARGET_MODEL_UPDATE_ITERATIONS, DISCOUNT_FACTOR


class DoubleDQNAgent(Agent):
    """
    Double Deep Q-Networks
    """

    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)
        self.target_model = self.build_model()
        self.iterations = 0

    def learn(self, batch):
        """
        Learning algorithm for DoubleDQN
        """
        self.iterations += 1

        # Update target network periodically
        if self.iterations == TARGET_MODEL_UPDATE_ITERATIONS:
            print('updating target model')
            model_weights = self.model.get_weights()
            self.target_model.set_weights(model_weights)
            self.iterations = 0

        states, actions, rewards, next_states, dones = list(map(np.array, list(zip(*batch))))

        # predict target q values using the target model
        next_q_values = self.target_model.predict(next_states)

        # Flip all True's to False's and False's to True's.
        is_not_done = np.logical_not(dones.reshape(len(batch), 1))

        target_q_values = rewards.reshape(len(batch), 1) + (next_q_values * DISCOUNT_FACTOR * is_not_done)

        loss = self.model.train_on_batch(states, target_q_values)
        return loss
