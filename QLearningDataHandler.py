import pickle

import matplotlib.pyplot as plt


class QLearningDataHandler:

    @staticmethod
    def plot_loss_curve(loss_values):
        x = [i for i in range(len(loss_values))]
        y = [loss for loss in loss_values]

        plt.xlabel('Steps')
        plt.ylabel('Loss')

        plt.plot(x, y)
        plt.show()

    @staticmethod
    def plot_reward_curve(reward_values):
        x = [i for i in range(len(reward_values))]
        y = [i for i in reward_values]

        plt.xlabel('Epoch')
        plt.ylabel('Total Reward')

        plt.plot(x, y)
        plt.show()

    @staticmethod
    def plot_time_taken_curve(time_values):
        x = [i for i in range(len(time_values))]
        y = [i for i in time_values]

        plt.xlabel('Epoch')
        plt.ylabel('Time taken (units)')

        plt.plot(x, y)
        plt.show()

    @staticmethod
    def plot_epsilon_curve(epsilon_values):
        x = [i for i in range(len(epsilon_values))]
        y = [epsilon for epsilon in epsilon_values]

        plt.xlabel('Epoch')
        plt.ylabel('Epsilon')

        plt.plot(x, y)
        plt.show()

    def load_data_from_file(self):
        with open('output/loss_values.txt', 'rb') as file:
            loss_values = pickle.load(file)
            self.plot_loss_curve(loss_values)

        with open('output/reward_values.txt', 'rb') as file:
            reward_values = pickle.load(file)
            self.plot_reward_curve(reward_values)

        with open('output/epsilon_values.txt', 'rb') as file:
            epsilon_values = pickle.load(file)
            self.plot_epsilon_curve(epsilon_values)

        with open('output/time_values.txt', 'rb') as file:
            time_values = pickle.load(file)
            self.plot_time_taken_curve(time_values)


# handler = QLearningDataHandler()
# handler.load_data_from_file()
