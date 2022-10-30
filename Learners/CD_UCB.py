import numpy as np
from Learners.Change_detect import Change_detect
from Learners.UCB import UCB


class CD_UCB(UCB):
    def __init__(self, n_arms, M=20, eps=0.05, h=40, alpha=0.01):
        super().__init__(n_arms)
        self.change_detection = [Change_detect(M, eps, h) for _ in range(n_arms)]
        self.valid_rewards_per_arms = [[] for _ in range(n_arms)]
        self.detections = [[] for _ in range(n_arms)]
        self.alpha = alpha

    def pull_arm(self, margin):
        if np.random.binomial(1, 1 - self.alpha):
            return super().pull_arm(margin)
        else:
            return np.random.randint(0, 4)

    def update(self, pulled_arm, reward, buyers, offers, margin):
        self.t += 1

        if self.change_detection[pulled_arm].update(buyers / offers * margin[pulled_arm]):
            self.detections[pulled_arm].append(self.t)
            self.valid_rewards_per_arms[pulled_arm] = []
            self.change_detection[pulled_arm].reset()
            self.total_views[pulled_arm] = 0
            self.empirical_means[pulled_arm] = 0
            # print("!!!!!!!!!!CHANGE DETECTED!!!!!!!", self.t, pulled_arm)

        self.total_views[pulled_arm] += offers
        # self.empirical_means[pulled_arm] = np.mean(self.valid_rewards_per_arms[pulled_arm])
        self.empirical_means[pulled_arm] = (self.empirical_means[pulled_arm] * (
                    self.total_views[pulled_arm] - offers) + buyers) / self.total_views[pulled_arm]

        self.update_observations(pulled_arm, reward, buyers, offers)

        for a in range(self.n_arms):
            # n_samples = len(self.rewards_per_arm[a])
            n_samples = self.total_views[a]
            self.confidence[a] = (2 * np.log(self.t) / n_samples) ** 0.5 if n_samples > 0 else np.inf

    def update_observations(self, pulled_arm, reward, buyers, offers):
        self.rewards_per_arm[pulled_arm].append(reward)

        # Use reward as reward
        # self.valid_rewards_per_arms[pulled_arm].append(reward)

        # Use conversion rate as reward
        # self.valid_rewards_per_arms[pulled_arm].append(buyers)

        self.collected_rewards = np.append(self.collected_rewards, reward)