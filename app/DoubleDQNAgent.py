from app.Agent import Agent


class DoubleDQNAgent(Agent):
    """
    Double Deep Q-Networks
    """

    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)

    def learn(self, batch):
        pass

