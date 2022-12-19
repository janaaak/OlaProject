import numpy as np
from Config import config3 as cf3, config1 as cf2, config as cf1

from Learners.SW_UCB import SW_UCB
from Learners.CD_UCB import CD_UCB
from Environment.simulator import Simulator


def step_6():
    #A 150 cambiano i CR ma la price config resta uguale
    #A 300 i CR restano uguali a quelli di 150 ma cambia la price config
    #Comprensibile il risultato dato che i learner scelgono gli arm in base alla stima che fanno del CR

    n_experiments = 5
    time_horizon = 450
    sim = [Simulator(i) for i in range(3)]
    cf=[cf1, cf2, cf3]
    rewardsSW_exp = []
    rewardsCD_exp = []

    for e in range(n_experiments):
        sw = [SW_UCB(sim[0].n_prices, 20) for i in range(sim[0].n_products)]
        cd = [CD_UCB(sim[0].n_prices) for i in range(sim[0].n_products)]

        print("Exp:", e)

        rewardsSW = np.array([])
        rewardsCD = np.array([])

        for t in range(time_horizon):
            if t<150:
                phase=0
            elif t<300:
                phase=1
            else:
                phase=2
            # TS Learners
            price_conf = np.array([sw[i].pull_arm(cf[phase].margin[i]) for i in range(sim[phase].n_products)])
            reward, buyers, offers, _, _, _, _ = sim[phase].simulate(price_conf)
            for p in range(sim[phase].n_products):
                sw[p].update(price_conf[p], reward[p], buyers[p], offers[p])
            rewardsSW = np.append(rewardsSW, np.sum(reward))
            if t==149:
                print("SW @150: ", price_conf)
            if t == 299:
                print("SW @300: ", price_conf)
            if t == 449:
                print("SW @450: ", price_conf)


        for t in range(time_horizon):
            if t<150:
                phase=0
            elif t<300:
                phase=1
            else:
                phase=2
            # UCB
            price_conf = np.array([cd[i].pull_arm(cf[phase].margin[i]) for i in range(sim[phase].n_products)])
            reward, buyers, offers, _, _, _, _ = sim[phase].simulate(price_conf)
            for p in range(sim[phase].n_products):
                cd[p].update(price_conf[p], reward[p], buyers[p], offers[p], cf[phase].margin[p])
            rewardsCD = np.append(rewardsCD, np.sum(reward))
            if t == 149:
                print("CD @150: ", price_conf)
            if t == 299:
                print("CD @300: ", price_conf)
            if t == 449:
                print("CD @450: ", price_conf)

        rewardsSW_exp.append(rewardsSW)
        rewardsCD_exp.append(rewardsCD)
    return rewardsSW_exp, rewardsCD_exp


