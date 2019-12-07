import pickle

import matplotlib.pyplot as plt
from scipy.signal import lfilter

blue = "#3355dd"
red = "#dd4411"


class QLearningDataHandler:

    def plot_rewards(self):
        # get reward values
        baseline_rewards, dqn_rewards, double_dqn_rewards = self.get_reward_values()

        # create axis
        x = [i for i in range(len(baseline_rewards))]
        y_baseline_rewards = [i for i in baseline_rewards]
        y_dqn_rewards = [i for i in dqn_rewards]
        y_double_dqn_rewards = [i for i in double_dqn_rewards]

        # remove noise from data
        n = 30
        yy_baseline_rewards = lfilter([1.0 / n] * n, 1, y_baseline_rewards)
        yy_dqn_rewards = lfilter([1.0 / n] * n, 1, y_dqn_rewards)
        yy_double_dqn_rewards = lfilter([1.0 / n] * n, 1, y_double_dqn_rewards)

        # plot curves
        plt.plot(x, yy_baseline_rewards, linewidth=0.5, c=red, label='Baseline')
        plt.plot(x, yy_dqn_rewards, linewidth=0.5, c=blue, label='DQN')
        plt.plot(x, yy_double_dqn_rewards, linewidth=0.5, c="green", label='DoubleDQN')

        # annotate plot
        plt.xlabel('Epoch')
        plt.ylabel('Total Reward')
        plt.title('Total Reward obtained per Epoch in Space Invaders')
        plt.legend(loc='best')
        plt.show()

    def plot_time_steps(self):
        # get time step values
        baseline_steps, dqn_steps, double_dqn_steps = self.get_time_steps()

        # create axis
        x = [i for i in range(len(baseline_steps))]
        y_baseline_steps = [i for i in baseline_steps]
        y_dqn_steps = [i for i in dqn_steps]
        y_double_dqn_steps = [i for i in double_dqn_steps]

        # remove noise from data
        n = 30
        yy_baseline_steps = lfilter([1.0 / n] * n, 1, y_baseline_steps)
        yy_dqn_steps = lfilter([1.0 / n] * n, 1, y_dqn_steps)
        yy_double_dqn_steps = lfilter([1.0 / n] * n, 1, y_double_dqn_steps)

        # plot curves
        plt.plot(x, yy_baseline_steps, linewidth=0.5, c=red, label='Baseline')
        plt.plot(x, yy_dqn_steps, linewidth=0.5, c=blue, label='DQN')
        plt.plot(x, yy_double_dqn_steps, linewidth=0.5, c="green", label='DoubleDQN')

        # annotate plot
        plt.xlabel('Epoch')
        plt.ylabel('Time taken (units)')
        plt.title('Time taken per Epoch in Space Invaders')
        plt.legend(loc='best')
        plt.show()

    def plot_loss_curve_dqn(self):
        # get loss values
        dqn_loss, _ = self.get_loss_values()

        # create axis
        x = [i for i in range(len(dqn_loss))]
        y_dqn_loss = [i for i in dqn_loss]

        # plot curve
        plt.plot(x, y_dqn_loss, c=blue, label='DQN')

        # annotate plot
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Loss at every time step in Space Invaders (DQN)')
        plt.legend(loc='best')
        plt.show()

    def plot_loss_curve_double_dqn(self):
        # get loss values
        _, double_dqn_loss = self.get_loss_values()

        # create axis
        x = [i for i in range(len(double_dqn_loss))]
        y_double_dqn_loss = [i for i in double_dqn_loss]

        # plot curve
        plt.plot(x, y_double_dqn_loss, linewidth=0.5, c="green", label='DoubleDQN')

        # annotate plot
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Loss at every time step in Space Invaders (DoubleDQN)')
        plt.legend(loc='best')
        plt.show()

    def plot_epsilon_curve(self):
        epsilon_values = self.get_epsilon_values()

        x = [i for i in range(len(epsilon_values))]
        y = [epsilon for epsilon in epsilon_values]

        plt.xlabel('Epoch')
        plt.ylabel('Epsilon')
        plt.title('Epsilon Curve')

        plt.plot(x, y)
        plt.show()

    @staticmethod
    def get_reward_values():
        baseline_rewards_dir = 'output/2019_12_05_12:15:00/reward_values_SpaceInvaders-ram-v0_Random.txt'
        dqn_rewards_dir = 'output/2019_12_05_07:14:44/reward_values_SpaceInvaders-ram-v0_DQNAgent.txt'
        double_dqn_rewards_dir = 'output/2019_12_05_08:25:54/reward_values_SpaceInvaders-ram-v0_DoubleDQNAgent.txt'

        baseline_rewards_file = open(baseline_rewards_dir, "rb")
        dqn_rewards_file = open(dqn_rewards_dir, "rb")
        double_dqn_rewards_file = open(double_dqn_rewards_dir, "rb")

        baseline_rewards = pickle.load(baseline_rewards_file)
        dqn_rewards = pickle.load(dqn_rewards_file)
        double_dqn_rewards = pickle.load(double_dqn_rewards_file)

        return baseline_rewards, dqn_rewards, double_dqn_rewards

    @staticmethod
    def get_time_steps():
        baseline_steps_dir = 'output/2019_12_05_12:15:00/time_values_SpaceInvaders-ram-v0_Random.txt'
        dqn_steps_dir = 'output/2019_12_05_07:14:44/time_values_SpaceInvaders-ram-v0_DQNAgent.txt'
        double_dqn_steps_dir = 'output/2019_12_05_08:25:54/time_values_SpaceInvaders-ram-v0_DoubleDQNAgent.txt'

        baseline_steps_file = open(baseline_steps_dir, "rb")
        dqn_steps_file = open(dqn_steps_dir, "rb")
        double_dqn_steps_file = open(double_dqn_steps_dir, "rb")

        baseline_steps = pickle.load(baseline_steps_file)
        dqn_steps = pickle.load(dqn_steps_file)
        double_dqn_steps = pickle.load(double_dqn_steps_file)

        return baseline_steps, dqn_steps, double_dqn_steps

    @staticmethod
    def get_loss_values():
        dqn_loss_dir = 'output/2019_12_05_07:14:44/loss_values_SpaceInvaders-ram-v0_DQNAgent.txt'
        double_dqn_loss_dir = 'output/2019_12_05_08:25:54/loss_values_SpaceInvaders-ram-v0_DoubleDQNAgent.txt'

        dqn_loss_file = open(dqn_loss_dir, "rb")
        double_dqn_loss_file = open(double_dqn_loss_dir, "rb")

        dqn_loss = pickle.load(dqn_loss_file)
        double_dqn_loss = pickle.load(double_dqn_loss_file)

        dqn_loss_file.close()
        double_dqn_loss_file.close()

        return dqn_loss, double_dqn_loss

    @staticmethod
    def get_epsilon_values():
        # epsilon decayed at the same rate for all the models
        with open(f'output/2019_12_05_08:25:54/epsilon_values_SpaceInvaders-ram-v0_DoubleDQNAgent.txt', 'rb') as file:
            return pickle.load(file)


handler = QLearningDataHandler()
handler.plot_rewards()
handler.plot_time_steps()
handler.plot_epsilon_curve()
handler.plot_loss_curve_dqn()
handler.plot_loss_curve_double_dqn()
