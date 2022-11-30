import random
from Config import config as cf


class UserClass:
    def __init__(self, conversion_rates, min_daily_users, max_daily_users, alphas, sold_items, graph_proba):
        self.conversion_rates = conversion_rates # conversion rate []
        self.min_daily_users = min_daily_users
        self.max_daily_users = max_daily_users
        self.alphas = alphas
        self.sold_items = sold_items
        self.graph_proba = graph_proba

    def get_daily_users(self):
        return random.randint(self.min_daily_users, self.max_daily_users)
    
    def get_sold_items(self):
        return random.randint(1, self.max_sold_items)