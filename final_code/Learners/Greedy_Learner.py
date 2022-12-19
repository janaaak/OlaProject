from Learners.Learner import Learner
import numpy as np

class Greedy_Learner(Learner):
    def __init__(self, n_arms, n_items):
        super().__init__(n_arms)
        self.n_items = n_items
        self.expected_rewards = np.zeros(n_arms)

    def pull_arm(self, id, price_conf):
        if self.t == 0:
            return np.zeros(self.n_items, dtype=np.int8)
        pulled_arms = price_conf.copy()
        if pulled_arms[id] < self.n_arms-1:
            pulled_arms[id] += 1
        return pulled_arms

    def update(self):
        self.t+=1
        #self.update_observations(pulled_arm, reward)
        #self.expected_rewards[pulled_arm] = (self.expected_rewards[pulled_arm]*(self.t-1) + reward)/self.t