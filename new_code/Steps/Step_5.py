import numpy as np
from Config import config as cf
from Learners.TS_Learner import TS_Learner
from Environment.simulator import Simulator


def step_5(time_horizon):
    n_experiments = 5
    #time_horizon = 300
    sim = Simulator(0)
    rewardsTS_exp = []
    rewardsUCB_exp = []

    for e in range(n_experiments):
        ts = [TS_Learner(sim.n_prices, cf.alphas_mean[i], cf.sold_items_mean[i]) for i in range(sim.n_products)]
        print("Exp:", e)

        rewardsTS = np.array([])

        for t in range(time_horizon):
            # TS Learners
            price_conf = np.array([ts[i].pull_arm_step5(cf.margin[i]) for i in range(sim.n_products)])
            reward, buyers, offers, alphas, items, history, previous = sim.simulate(price_conf)
            graph_prob = sim.estimate_probabilities(history, previous)
            # print(graph_prob)
            for p in range(sim.n_products):
                ts[p].update(price_conf[p], reward[p], buyers[p], offers[p], graph=graph_prob[p])
            rewardsTS = np.append(rewardsTS, np.sum(reward))

        print("TS final price conf: ", price_conf)
        rewardsTS_exp.append(rewardsTS)
    return rewardsTS_exp