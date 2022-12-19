import math
from Learners.Learner import Learner
import numpy as np

class TS_Learner(Learner):

    def __init__(self, n_arms, alpha=None, items=None, graph=None):
        super().__init__(n_arms)
        self.beta_parameters = np.ones((n_arms,2))
        self.total_offers = np.zeros(n_arms)

        if np.all(alpha != None): 
            self.alpha = alpha
        else:
            self.alpha = 1

        if np.all(items != None):
            self.items = items
        else:
            self.items = 1

        if np.all(graph != None):
            self.graph = graph
        else:
            self.graph = np.ones((n_arms, 5))

        self.arm_counter = np.zeros(n_arms)

    def pull_arm(self, margin):
        b = np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1])

        #print(b*margin*self.alpha*self.items*np.sum(self.graph, axis=1))
        idx = np.argmax(b*margin*self.alpha*self.items)

        return idx

    def pull_arm_step5(self, margin):
        b = np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1])

        idx = np.argmax(margin*np.sum(self.graph, axis=1))

        return idx


    def update(self, pulled_arm, reward, buyers, offers, alpha=None, items=None, graph=None):
        self.t+=1
        self.total_offers[pulled_arm] += offers
        self.update_observations(pulled_arm, reward)

        self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + buyers.astype(int)
        self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + offers.astype(int) - buyers.astype(int)
        #print("pulled_arm", pulled_arm)
        #print(self.beta_parameters) 

        if alpha != None:
            self.alpha = (self.alpha * (self.t-1) + alpha)/self.t
        
        if items != None and items != 0 and not math.isnan(items):
            self.items = (self.items * (self.t-1) + items)/self.t
           
        
        if np.all(graph != None):
            self.arm_counter[pulled_arm] += 1
            self.graph[pulled_arm] = (self.graph[pulled_arm]*(self.arm_counter[pulled_arm] - 1) + graph)/self.arm_counter[pulled_arm]


    def plot_distribution(self):
        from scipy.stats import beta
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1)

        colors=['r', 'g', 'b', 'm']
        for i in range(self.n_arms):
            a1, b1 = self.beta_parameters[i, 0], self.beta_parameters[i, 1]
            #mean, var, skew, kurt = beta.stats(a, b, moments='mvsk')
            x = np.linspace(beta.ppf(0.01, a1, b1),
                    beta.ppf(0.99, a1, b1), 100)
            ax.plot(x, beta.pdf(x, a1, b1),
                colors[i], lw=0.5, alpha=0.6, label='beta pdf')