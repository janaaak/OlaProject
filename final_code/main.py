from Environment.simulator import *
from Steps.Step_2 import step_2
from Steps.Step_3 import step_3
from Steps.Step_4 import step_4
from Steps.Step_5 import step_5
from Steps.Step_6 import step_6
from Steps.Step_7 import step_7
import matplotlib.pyplot as plt

from Config import config as cf

def main():
    sim = Simulator(0)
    time_horizon = 300
    while True:
        step = int(input("Choose a step [2-3-4-5-6-7]: "))

        if step == 2:
            opt, _, best_price_conf = sim.bruteforce()
            opt_per_product, max_price_conf = step_2()
            print("\nTotal revenue from the brute force algorithm:", opt,
                  "\nOptimal price configuration used", best_price_conf)
            print("\nTotal revenue from the greedy algorithm:", np.sum(opt_per_product),
                  "\nOptimal price configuration used", max_price_conf)
            break

        elif step == 3:
            opt, opt_per_product, best_price_conf = sim.bruteforce()
            rewardsTS, rewardsUCB, mean_rewards = step_3(time_horizon)
            bound = compute_UCBbound(opt_per_product, mean_rewards, time_horizon)
            print(bound)
            plot_regret(opt, rewardsTS, rewardsUCB, time_horizon, bound=300)
            plot_reward(opt, rewardsTS, rewardsUCB, time_horizon)
            print("The theoretical bound of UCB regret over a time horizon of ", time_horizon, " days, is ", bound)
            break

        elif step == 4:
            opt, opt_per_product, best_price_conf = sim.bruteforce()
            rewardsTS, rewardsUCB = step_4(time_horizon)
            plot_regret(opt, rewardsTS, rewardsUCB, time_horizon, bound=0)
            plot_reward(opt, rewardsTS, rewardsUCB, time_horizon)
            break

        elif step == 5:
            opt, opt_per_product, best_price_conf = sim.bruteforce()
            rewardsTS = step_5(time_horizon)
            plot_regret(opt, rewardsTS, None, time_horizon, step=5)
            plot_reward(opt, rewardsTS, None, time_horizon, step=5)
            break

        elif step == 6:
            opt_final = np.array([])
            sim1 = Simulator(1)
            sim2 = Simulator(2)
            opt, opt_per_product, best_price_conf = sim.bruteforce()
            opt1, opt_per_product1, best_price_conf = sim1.bruteforce()
            opt2, opt_per_product2, best_price_conf = sim2.bruteforce()

            for i in range(0, 150):
                opt_final = np.append(opt_final, opt)
            for i in range(150, 300):
                opt_final = np.append(opt_final, opt1)
            for i in range(300, 450):
                opt_final = np.append(opt_final, opt2)
            rewardsSW, rewardsCD = step_6()
            plot_regret(opt_final, rewardsSW, rewardsCD, 450, step=6)
            plot_reward(opt_final, rewardsSW, rewardsCD, 450, step=6)
            break

        elif step == 7:
            opt, opt_per_product, best_price_conf = sim.bruteforce()
            rewardsTS, rewardsUCB = step_7(time_horizon)
            plot_regret(opt, rewardsTS, rewardsUCB, time_horizon)
            plot_reward(opt, rewardsTS, rewardsUCB, time_horizon)
            break

        else:
            print("You entered an invalid step number. Please try again.")


def plot_regret(opt, rewardsTS_exp, rewardsUCB_exp, time_horizon, bound=0, step=0):
    plt.figure(0)
    if step == 6:
        labels = ["SW UCB", "CD UCB"]
        plt.axvline(x=150)
        plt.axvline(x=300)
    elif step == 5:
        labels = ["Learners"]
    else:
        labels = ["TS", "UCB", "Bound"]
    plt.xlabel("t")
    plt.ylabel("Regret")
    plt.plot(np.cumsum(np.mean(opt - rewardsTS_exp, axis=0)) , 'r', label=labels[0])
    if rewardsUCB_exp is not None:
        plt.plot(np.cumsum(np.mean(opt - rewardsUCB_exp, axis=0)), 'g', label=labels[1])
    if bound != 0:
        plt.plot(time_horizon * [bound], 'b', label=labels[2])

    x = np.arange(time_horizon)
    y_ts = (np.cumsum(np.mean(opt - rewardsTS_exp, axis=0)))
    if rewardsUCB_exp is not None:
        y_ucb = (np.cumsum(np.mean(opt - rewardsUCB_exp, axis=0)))

    dev_ts = np.std(np.cumsum(opt - rewardsTS_exp, axis=1), axis=0)
    if rewardsUCB_exp is not None:
        dev_ucb = np.std(np.cumsum(opt - rewardsUCB_exp, axis=1), axis=0)

    n_ts = len(rewardsTS_exp)
    if rewardsUCB_exp is not None:
        n_ucb = len(rewardsUCB_exp)

    plt.fill_between(x, y_ts - dev_ts * 1.96 / np.sqrt(n_ts), y_ts + dev_ts * 1.96 / np.sqrt(n_ts), color='r',
                     alpha=0.4)
    if rewardsUCB_exp is not None:
        plt.fill_between(x, y_ucb - dev_ucb * 1.96 / np.sqrt(n_ucb), y_ucb + dev_ucb * 1.96 / np.sqrt(n_ucb), color='g',
                         alpha=0.4)

    plt.legend()
    plt.show()


def plot_reward(opt, rewardsTS_exp, rewardsUCB_exp, time_horizon, step=0):
    plt.figure(0)
    plt.xlabel("t")
    plt.ylabel("Reward")

    if step == 5:
        labels=["Learners"]
        plt.plot(time_horizon * [opt], 'b', label='Optimal')
    elif step == 6:
        labels = ["SW UCB", "CD UCB"]
        plt.plot(opt, 'b', label='Optimal')
        plt.axvline(x=150)
        plt.axvline(x=300)
    else:
        labels = ["TS", "UCB"]
        plt.plot(time_horizon * [opt], 'b', label='Optimal')




    window = 10

    average_y = moving_average(np.mean(rewardsTS_exp, axis=0), window)
    plt.plot(average_y[:-10], 'y', label=labels[0])
    # plt.plot(np.mean(rewardsTS_exp, axis=0),'r', label='TS')

    if rewardsUCB_exp is not None:
        average_y = moving_average(np.mean(rewardsUCB_exp, axis=0), window)
        plt.plot(average_y[:-10], 'c', label=labels[1])
    # plt.plot(np.mean(rewardsUCB_exp, axis=0),'g', label='UCB')

    plt.legend()
    plt.show()


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w


def compute_UCBbound(opt_per_product, mean_rewards, time_horizon):
    s = 0
    for i in range(5):
        for j in range(4):
            delta = opt_per_product[i] - mean_rewards[i][j]
            if delta > 0:
                s += (4 * np.log(time_horizon) / delta + 8 * delta)
    return s


if __name__ == "__main__":
    main()
