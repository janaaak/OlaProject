from Learners.UCB import UCB
import numpy as np

class SW_UCB(UCB):
    def __init__(self, n_arms, window_size=20, alpha=None, items=None, graph=None):
        super().__init__(n_arms)
        self.empirical_means = np.zeros(n_arms)
        self.confidence = np.array([np.inf]*n_arms)
        self.window_size = window_size
        self.pulled_arms = [[] for i in range(n_arms)]

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
        upper_conf = self.empirical_means + self.confidence
        return np.argmax(upper_conf*margin)

    def update(self, pulled_arm, reward, buyers, offers, alpha=None, items=None, graph=None):
        self.t +=1
        self.update_observations(pulled_arm,reward)
        self.pulled_arms[pulled_arm].append([buyers.astype(int), offers.astype(int)])
        
        for arm in range(self.n_arms):
            # Mean of rewards
            # if len(self.rewards_per_arm[arm]):
            #     self.empirical_means[arm] = 0
            # else:
            #     self.empirical_means[arm] = np.mean(self.rewards_per_arm[arm][-self.window_size:])

            # Mean of conversion rates

            self.empirical_means[arm] = sum(el[0] for el in self.pulled_arms[arm][-self.window_size:])
            self.empirical_means[arm] /= sum(el[1] for el in self.pulled_arms[arm][-self.window_size:])

            #n_samples = np.sum(self.pulled_arms[-self.window_size: , 0] == arm)
            n_samples = sum(el[1] for el in self.pulled_arms[arm][-self.window_size:])
            self.confidence[arm] = (2*np.log(self.t)/n_samples)**0.5 if n_samples>0 else np.inf
        

        if alpha != None:
            self.alpha = (self.alpha * (self.t-1) + alpha)/self.t
        
        if items != None:
            self.items = (self.items * (self.t-1) + items)/self.t
        
        if np.all(graph != None):
            self.arm_counter[pulled_arm] += 1
            self.graph[pulled_arm] = (self.graph[pulled_arm]*(self.arm_counter[pulled_arm] - 1) + graph)/self.arm_counter[pulled_arm]

    
    def plot_distribution(self):
        from scipy.stats import beta
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1)

        colors=['r', 'g', 'b', 'm']
        x = [0, 1, 2, 3]
        #y = [(i,j) for i,j in zip(self.empirical_means, self.empirical_means+self.confidence)]
        #print(y)
        plt.plot((x,x), (self.empirical_means, self.empirical_means+self.confidence))