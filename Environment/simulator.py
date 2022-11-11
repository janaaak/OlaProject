"""
Environment: Simulator
"""
from copy import deepcopy

import numpy as np
from Environment.user_class import UserClass
from Data import config as cf
import matplotlib.pyplot as plt

# np.random.seed(seed=3259414887)
# random.seed(1111)

T = 100

class Simulator():
    n_products = 5
    n_prices = 4
    l = 0.8

    def __init__(self):

        self.cr_mean = cf.cr_mean
        self.alphas_mean = cf.alphas_mean
        self.margin = cf.margin
        self.graph_proba_mean = cf.graph_proba_mean
        self.daily_users = (int)(np.random.normal(400, 10))
        self.uS = UserClass(
            cf.conversion_rates_S,
            cf.alphas_S,
            cf.sold_items_S,
            cf.graph_proba_S,
            self.daily_users
        )

        self.uM = UserClass(
            cf.conversion_rates_M,
            cf.alphas_M,
            cf.sold_items_M,
            cf.graph_proba_M,
            self.daily_users
        )

        self.uF = UserClass(
            cf.conversion_rates_F,
            cf.alphas_F,
            cf.sold_items_F,
            cf.graph_proba_F,
            self.daily_users
        )

        self.user_classes = [self.uS, self.uM, self.uF]
        self.sec_prod = cf.sec_prod

    def dec_to_base(self, num, base=4):  # Maximum base - 36
        base_num = ""
        while num > 0:
            dig = int(num % base)
            if dig < 10:
                base_num += str(dig)
            else:
                base_num += chr(ord('A') + dig - 10)  # Using uppercase letters
            num //= base

        base_num = base_num[::-1]  # To reverse the string

        return np.array([int(a) for a in str(base_num).zfill(self.n_products)])

    def rewards_per_class(self, user_c):
        total_rewards = np.zeros(5)
        prod_reward = np.zeros((self.n_products, self.n_prices))
        total_graph_weights = np.ones(self.n_products)
        for p in range(self.n_products):
            for i in range(self.n_products):
                if user_c.graph_proba[p][i] != 0:           #not on the diagonal (=0)
                    total_graph_weights[p] *= user_c.graph_proba[p][i]      #multiply the rows of the graph proba as we move from product to product
                prod_reward[p] += cf.margin[p] * user_c.conversion_rates[p] * user_c.user_num[p + 1] * total_graph_weights[p] * user_c.get_sold_items()
                                #margin * conv rate * number of users for this product * graph weights until now * the number of sold items per class
        for n in range(self.n_products):
            for m in range(self.n_prices):
                total_rewards [n] += prod_reward[n][m]

                #get total rewards per product

        return total_rewards


    def Greedy_alg(self, users):
        price_index = [0, 0, 0, 0, 0]       #initial price config
        prev = 0
        price_hist = []
        reward_hist = []
        max_reward = np.sum(self.rewards_per_class(users))          #total reward from all products
        bool_max_prices = np.zeros(5)                               #chosen price config
        it = 0

        while max_reward > prev and np.sum(bool_max_prices) < 5:
            if it > 0:
                price_index[np.argmax(sum_rewards)] += 1        #to choose which config gets the max price
            reward_hist.append(max_reward)                      #total rewards until now
            price_hist.append(deepcopy(price_index))            #price config update

            prev = max_reward
            sum_rewards = np.zeros(5)

            for i in range(self.n_products):
                if price_index[i] < 3:
                    price_index[i]+=1           #we check next price config
                    sum_rewards[i] = np.sum(self.rewards_per_class(users))          #we get rewards of it
                    price_index[i] -=1              #then return to initial
                else:
                    bool_max_prices[i] = True           #then we are done with this product
            max_reward = np.max(sum_rewards)            #get the max
            if max_reward > prev:                       # we move to next iteration
                it+=1

        return price_hist, reward_hist, it


    def bruteforce(self, step=0):
        max = 0
        best_conf = None
        for i in range(self.n_prices ** self.n_products):
            conf = self.dec_to_base(i)
            reward = 0
            #           if step==5:
            #               _, buyers, views, alphas, items, history, previous = self.simulate(conf)
            #                graph_prob = self.estimate_probabilities(history, previous)
            #            #print(i)
            for item, p in enumerate(conf):
                #            if step == 5:
                #                   reward += self.cr_mean[item][p] * self.margin[item][p] * np.sum(graph_prob[item])
                #                else :
                reward += self.alphas_mean[item+1] * self.cr_mean[item][p] * self.margin[item][p]

                if reward > max:
                    max = reward
                    best_conf = conf

        product_revenue = np.zeros(self.n_products)
        for i in range(100):
            # print("Bruteforce:", i)
            reward = self.simulate(best_conf, users=100)[0]
            product_revenue += reward
            product_revenue /= 100

        revenue = np.sum(product_revenue)
        print(f"Bruteforce\n Max reward: {revenue}\n Best configuration: {best_conf}")
        return revenue, product_revenue, best_conf

    #    only for step 5
    #    def estimate_probabilities(self, dataset, previous):

    #      credits = np.zeros((self.n_products, self.n_products))
    #      active = np.zeros(self.n_products)

    #      for episode, prev in zip(dataset, previous):
    #       for e, p in zip(episode, prev):
    #         idx = np.argwhere(e).reshape(-1)
    #         active[idx] += 1
    #         if p >= 0:
    #           credits[p, idx] += 1
    #         else:
    #           credits[idx, idx] += 0
    #      # print(credits)
    #      # print(active)
    #      # print(np.sum(credits, axis=0))
    #      for i in range(self.n_products):
    #       for j in range(self.n_products):
    #         credits[i, j] = credits[i, j] / active[i]
    #      # print(credits.T)
    #      return credits * self.lam

    def initial_node(self, alphas):
        nodes = np.array(range(self.n_products + 1))
        initial_node = np.random.choice(nodes, 1, p=alphas)
        initial_active_node = np.zeros(self.n_products + 1)
        initial_active_node[initial_node] = 1
        initial_active_node = initial_active_node[1:]
        return initial_active_node

    def simulate(self, price_conf, users=None, cl_number=-1):
        reward = np.zeros(self.n_products)
        buyers = np.zeros(self.n_products)
        views = np.zeros(self.n_products)
        alphas = np.zeros(self.n_products)
        items = np.zeros(self.n_products)
        total_history = []
        total_previous = []

        total_users = 0
        daily_users = 0
        for cl_id, cl in enumerate(self.user_classes):
            if cl_number >= 0:
                if cl_id != cl_number:
                    continue
                if users == None:
                    daily_users = cl.daily_users
                else:
                    daily_users = users
            total_users = total_users + daily_users

            for i in range(daily_users):
                initial_active_node = self.initial_node(cl.alphas)
                if all(initial_active_node == 0):
                    continue

                alphas[np.argwhere(initial_active_node).reshape(-1)] += 1
                prob_matrix = cl.graph_probs.copy() * self.lam
                np.fill_diagonal(prob_matrix, 0)

                history = np.empty((0, self.n_products))
                active_nodes_list = np.array([initial_active_node])
                previous_all = np.zeros(self.n_products, dtype=np.int8) - 2
                previous_all[np.argwhere(initial_active_node).reshape(-1)] = -1

                t = 0
                while (len(active_nodes_list) > 0):
                    active_node = active_nodes_list[0].copy()
                    active_nodes_list = active_nodes_list[1:]
                    idx_active = np.argwhere(active_node).reshape(-1)

                    views[idx_active] += 1

                    if np.random.uniform(0, 1) < cl.conversion_rates[idx_active, price_conf[idx_active]]:

                        buyers[idx_active] += 1
                        items_sold = 1 + np.random.poisson(cl.sold_items[idx_active])
                        items[idx_active] += items_sold

                        reward[idx_active] += self.margin[idx_active,
                                                          price_conf[idx_active]] * items_sold

                        # we are saving in an array the probas of going from the active nodes to all the others
                        p = (prob_matrix.T * active_node).T
                        # to extract the indexes of the nodes where the activation proba is > 0
                        next_nodes = np.argwhere(p)

                        # returns 1 if edge is activated 0 otherwise
                        activated_edges = p > np.random.rand(p.shape[0], p.shape[1])
                        prob_matrix[:, idx_active] = 0

                        # if we have an active edge, the activated nodes are flipped
                        newly_active_nodes = (np.sum(activated_edges, axis=0) > 0) * (1 - active_node)

                        for idx in next_nodes:
                            if newly_active_nodes[idx] == 1:
                                prob_matrix[:, idx] = 0
                                a = np.zeros(5)
                                a[idx] = 1
                                active_nodes_list = np.concatenate(
                                    (active_nodes_list, [a]), axis=0)
                                previous_all[idx] = idx_active
                    # print(active_nodes_list)
                    history = np.concatenate((history, [active_node]), axis=0)

                previous = np.array([], dtype=np.int8)
                for e in history:
                    previous = np.append(
                        previous, previous_all[np.argwhere(e).reshape(-1)])

                total_history.append(history)
                total_previous.append(previous)

            items_mean = np.zeros(self.n_products)
            for i in range(self.n_products):
                if buyers[i] != 0:
                    items_mean[i] = items[i] / buyers[i]

            return reward / total_users, buyers, views, alphas / total_users, items_mean, total_history, total_previous

        # if __name__ == '__main__':
        #     sim = Simulator()
        #     revenue, max_price_conf = sim.bruteforce()
        #     product_revenue, max_price_conf = sim.step2()
        #     # rewardsTS_exp, rewardsUCB_exp = sim.step3()
        #     # rewardsTS_exp, rewardsUCB_exp = sim.step4()
        #     # rewardsTS_exp, rewardsUCB_exp = sim.step5()
        #     # rewardsTS_exp, rewardsUCB_exp = sim.step6()
        #     # print(rewardsTS_exp)
        #
        #     revenue = np.sum(product_revenue)
        #     print("Optimal is", revenue)
        #     print("Max price conf", max_price_conf)
        #     plt.figure(0)
        #     plt.xlabel("t")
        #     plt.ylabel("Regret")
        #     plt.plot(T * [revenue])
        #     # plt.plot(np.mean(rewardsTS_exp, axis=0),'r')
        #     # plt.plot(np.mean(rewardsUCB_exp, axis=0),'g')
        #     plt.plot(np.cumsum(T * [revenue]), 'b')
        #     # plt.plot(np.cumsum(np.mean(revenue - rewardsTS_exp, axis=0)), 'r')
        #     # plt.plot(np.cumsum(np.mean(revenue - rewardsUCB_exp, axis=0)), 'g')
        #     # plt.plot(np.cumsum(100*[revenue]-rewards_per_experiment))
        #     plt.show()

        # if __name__ == "__main__":
        #     results = Greedy_alg(self.user_classes[0])
        #     print(results[0])
        #     print(results[1])
        #     print(results[2])




