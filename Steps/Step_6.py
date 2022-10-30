import numpy as np
from Data import config3 as cf3, config2 as cf2, config as cf1

from Learners.SW_UCB import SW_UCB
from Learners.CD_UCB import CD_UCB
from Environment.simulator import Simulator


def step_6():
    #A 150 cambiano i CR ma la price config resta uguale
    #A 300 i CR restano uguali a quelli di 150 ma cambia la price config
    #Comprensibile il risultato dato che i learner scelgono gli arm in base alla stima che fanno del CR

    n_experiments = 5
    time_horizon = 540
    sim = [Simulator() for i in range(3)]
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
            if t<180:
                phase=0
            elif t<360:
                phase=1
            else:
                phase=2
            # UCB
            price_conf = np.array([cd[i].pull_arm(cf[phase].margin[i]) for i in range(sim[phase].n_products)])
            reward, buyers, offers, _, _, _, _ = sim[phase].simulate(price_conf)
            for p in range(sim[phase].n_products):
                cd[p].update(price_conf[p], reward[p], buyers[p], offers[p], cf[phase].margin[p])
            rewardsCD = np.append(rewardsCD, np.sum(reward))
            if t == 179:
                print("CD @180: ", price_conf)
            if t == 359:
                print("CD @360: ", price_conf)
            if t == 539:
                print("CD @540: ", price_conf)

        rewardsSW_exp.append(rewardsSW)
        rewardsCD_exp.append(rewardsCD)
    return rewardsSW_exp, rewardsCD_exp