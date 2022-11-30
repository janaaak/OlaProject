import numpy as np
from Config import config as cf
from Learners.TS_Learner import TS_Learner
from Learners.UCB import UCB
from Environment.simulator import Simulator


def step_7(time_horizon):
    n_experiments = 5
    #time_horizon = 100
    sim = Simulator(0)

    rewardsTS_exp = []
    rewardsUCB_exp = []

    for e in range(n_experiments):
        ts = [[TS_Learner(sim.n_prices) for i in range(sim.n_products)] for j in range(3)]
        ucb = [[UCB(sim.n_prices) for i in range(sim.n_products)] for j in range(3)]

        print("Exp:", e)

        rewardsTS = np.array([])
        rewardsUCB = np.array([])

        for cl in range(3):
            for t in range(time_horizon):
                # TS Learners
                price_conf = np.array([ts[cl][i].pull_arm(cf.margin[i]) for i in range(sim.n_products)])
                reward, buyers, offers, alphas, items ,_ ,_ = sim.simulate(price_conf, cl_number=cl)
                for p in range(sim.n_products):
                    ts[cl][p].update(price_conf[p], reward[p], buyers[p], offers[p], alphas[p], items[p])
                rewardsTS = np.append(rewardsTS, np.sum(reward))

            print("TS class: ", cl, "final price configuration :", price_conf)
        rewardsTS = np.mean(np.reshape(rewardsTS, (3, -1)), axis=0)
        
        for cl in range(3):
            for t in range(time_horizon):
                # UCB
                price_conf = np.array([ucb[cl][i].pull_arm(cf.margin[i]) for i in range(sim.n_products)])
                reward, buyers, offers, alphas, items ,_ ,_ = sim.simulate(price_conf, cl_number=cl)
                for p in range(sim.n_products):
                    ucb[cl][p].update(price_conf[p], reward[p], buyers[p], offers[p], alphas[p], items[p])
                rewardsUCB = np.append(rewardsUCB, np.sum(reward))
            print("UCB class: ", cl, "final price configuration :", price_conf)

        rewardsUCB = np.mean(np.reshape(rewardsUCB, (3, -1)), axis=0)

        rewardsTS_exp.append(rewardsTS)
        rewardsUCB_exp.append(rewardsUCB)

    return rewardsTS_exp, rewardsUCB_exp