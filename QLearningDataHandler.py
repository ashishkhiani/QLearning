import statistics

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
        y = [statistics.mean(i) for i in reward_values]

        plt.xlabel('Epoch')
        plt.xticks(x)
        plt.ylabel('Mean Reward')

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
