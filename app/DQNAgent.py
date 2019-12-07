import numpy as np
from app.Agent import Agent
from parameters import DISCOUNT_FACTOR


class DQNAgent(Agent):
    """
    Deep Q-Networks
    """

    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)

    def learn(self, batch):
        """
        Learning algorithm for DQN
        """
        states, actions, rewards, next_states, dones = list(map(np.array, list(zip(*batch))))

        # predict target q values 
        next_q_values = self.model.predict(next_states)

        is_not_done = np.logical_not(dones.reshape(len(batch), 1))  # Flip all Ts to Fs and Fs to Ts.
        target_q_values = rewards.reshape(len(batch), 1) + (next_q_values * DISCOUNT_FACTOR * is_not_done)

        loss = self.model.train_on_batch(states, target_q_values)
        return loss
