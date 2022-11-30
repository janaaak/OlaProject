import numpy as np
from Config import config as cf
from Learners.TS_Learner import TS_Learner
from Learners.UCB import UCB
from Environment.simulator import Simulator


def step_4(time_horizon):
    n_experiments = 5
    #time_horizon = 100
    sim = Simulator(0)

    rewardsTS_exp = []
    rewardsUCB_exp = []

    for e in range(n_experiments):
        ts = [TS_Learner(sim.n_prices) for i in range(sim.n_products)]
        ucb = [UCB(sim.n_prices) for i in range(sim.n_products)]

        print("Exp:", e)

        rewardsTS = np.array([])
        rewardsUCB = np.array([])

        for t in range(time_horizon):
            # TS Learners
            price_conf = np.array([ts[i].pull_arm(cf.margin[i]) for i in range(sim.n_products)])
            reward, buyers, offers, alphas, items ,_ ,_ = sim.simulate(price_conf)
            for p in range(sim.n_products):
                ts[p].update(price_conf[p], reward[p], buyers[p], offers[p], alphas[p], items[p])
            rewardsTS = np.append(rewardsTS, np.sum(reward))
        print("TS final price configuration: ", price_conf)

        for t in range(time_horizon):
            # UCB
            price_conf = np.array([ucb[i].pull_arm(cf.margin[i]) for i in range(sim.n_products)])
            reward, buyers, offers, alphas, items ,_ ,_ = sim.simulate(price_conf)
            for p in range(sim.n_products):
                ucb[p].update(price_conf[p], reward[p], buyers[p], offers[p], alphas[p], items[p])
            rewardsUCB = np.append(rewardsUCB, np.sum(reward))
        print("UCB final price configuration: ", price_conf)


        rewardsTS_exp.append(rewardsTS)
        rewardsUCB_exp.append(rewardsUCB)

    return rewardsTS_exp, rewardsUCB_exp