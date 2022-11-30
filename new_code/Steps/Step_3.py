import numpy as np
from Config import config as cf
from Learners.TS_Learner import TS_Learner
from Learners.UCB import UCB
from Environment.simulator import Simulator


def step_3(time_horizon):
    n_experiments = 5
    #time_horizon = 300
    sim = Simulator(0)
    rewardsTS_exp = []
    rewardsUCB_exp = []

    for e in range(n_experiments):
        # learners = [TS_Learner(sim.n_prices) for i in range(sim.n_products)]
        ts = [TS_Learner(sim.n_prices, cf.alphas_mean[i], cf.sold_items_mean[i]) for i in range(sim.n_products)]
        ucb = [UCB(sim.n_prices, cf.alphas_mean[i], cf.sold_items_mean[i]) for i in range(sim.n_products)]

        print("Exp:", e)

        rewardsTS = np.array([])
        rewardsUCB = np.array([])

        for t in range(time_horizon):
            # TS Learners
            price_conf = np.array([ts[i].pull_arm(cf.margin[i]) for i in range(sim.n_products)])
            reward, buyers, offers, _, _, _, _ = sim.simulate(price_conf)
            for p in range(sim.n_products):
                ts[p].update(price_conf[p], reward[p], buyers[p], offers[p])
            rewardsTS = np.append(rewardsTS, np.sum(reward))
        print("TS final price configuration: ", price_conf)
        print("TS reward: ", rewardsTS[-1])

        for t in range(time_horizon):
            # UCB
            price_conf = np.array([ucb[i].pull_arm(cf.margin[i]) for i in range(sim.n_products)])
            reward, buyers, offers, _, _, _, _ = sim.simulate(price_conf)
            for p in range(sim.n_products):
                ucb[p].update(price_conf[p], reward[p], buyers[p], offers[p])
            rewardsUCB = np.append(rewardsUCB, np.sum(reward))
        print("UCB final price configuration: ", price_conf)
        print("UCB reward: ", rewardsUCB[-1])

        rewardsTS_exp.append(rewardsTS)
        rewardsUCB_exp.append(rewardsUCB)
        mean_rewards = [ucb[i].get_mean_reward_per_arm() for i in range(sim.n_products)]
    return rewardsTS_exp, rewardsUCB_exp, mean_rewards