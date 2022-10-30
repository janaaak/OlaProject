"""
Environment: User Class
"""
import numpy as np
import random

class UserClass():
    def __init__(self, conversion_rates, alphas, sold_items, graph_proba, n_users):
        self.conversion_rates = conversion_rates  # conversion rate []
        self.alphas = alphas
        self.sold_items = sold_items
        self.graph_proba = graph_proba
        self.max_sold_items = np.max(sold_items)
        self.n_users = n_users
        self.user_num = np.around(self.n_users * self.alphas)

    def get_sold_items(self):
        return random.randint(1, (int)(self.max_sold_items))