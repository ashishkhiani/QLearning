import pickle

import matplotlib.pyplot as plt
from scipy.signal import lfilter

from parameters import EMULATION

dark_blue = "#99abee"
light_blue = "#3355dd"

light_red = "#eea28d"
dark_red = "#dd4411"


class QLearningDataHandler:

    @staticmethod
    def plot_loss_curve(loss_values):
        x = [i for i in range(len(loss_values))]
        y = [i for i in loss_values]

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

        plt.plot(x, y, c=dark_blue)

        n = 15
        b = [1.0 / n] * n
        a = 1
        yy = lfilter(b, a, y)
        plt.plot(x, yy, linewidth=2, linestyle="-", c=light_blue)

        plt.show()

    @staticmethod
    def plot_time_taken_curve(time_values):
        x = [i for i in range(len(time_values))]
        y = [i for i in time_values]

        plt.xlabel('Epoch')
        plt.ylabel('Time taken (units)')

        plt.plot(x, y, c=dark_blue)

        n = 15
        b = [1.0 / n] * n
        a = 1
        yy = lfilter(b, a, y)
        plt.plot(x, yy, linewidth=2, linestyle="-", c=light_blue)

        plt.show()

    @staticmethod
    def plot_epsilon_curve(epsilon_values):
        x = [i for i in range(len(epsilon_values))]
        y = [epsilon for epsilon in epsilon_values]

        plt.xlabel('Epoch')
        plt.ylabel('Epsilon')

        plt.plot(x, y)
        plt.show()

    def load_data_from_file(self, directory, _class):
        misc = f'{EMULATION}_{_class}'

        with open(f'{directory}/loss_values_{misc}.txt', 'rb') as file:
            loss_values = pickle.load(file)
            if loss_values:
                self.plot_loss_curve(loss_values)

        with open(f'{directory}/reward_values_{misc}.txt', 'rb') as file:
            reward_values = pickle.load(file)
            if reward_values:
                self.plot_reward_curve(reward_values)

        with open(f'{directory}/epsilon_values_{misc}.txt', 'rb') as file:
            epsilon_values = pickle.load(file)
            if epsilon_values:
                self.plot_epsilon_curve(epsilon_values)

        with open(f'{directory}/time_values_{misc}.txt', 'rb') as file:
            time_values = pickle.load(file)
            if time_values:
                self.plot_time_taken_curve(time_values)


handler = QLearningDataHandler()
handler.load_data_from_file('output/2019_12_05_12:15:00', 'Random')

