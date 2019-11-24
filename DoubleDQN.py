class DoubleDQN:
    """
    Double Deep Q-Networks
    """

    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

    def get_action(self):
        # TODO (currently returns random action)
        return self.action_space.sample()
