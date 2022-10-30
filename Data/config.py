"""
Configuration 1
"""
import numpy as np

#Students
conversion_rates_S = np.array([[0.92, 0.85, 0.78, 0.67],
                               [0.27, 0.19, 0.13, 0.05],
                               [0.56, 0.49, 0.42, 0.37],
                               [0.7 , 0.65, 0.6 , 0.55],
                               [0.53, 0.41, 0.35, 0.27]])
#Male adults
conversion_rates_M = np.array([[0.2 , 0.15, 0.1 , 0.05],
                               [0.87, 0.74, 0.69, 0.61],
                               [0.56, 0.49, 0.42, 0.37],
                               [0.69, 0.6 , 0.53, 0.45],
                               [0.95, 0.83, 0.77, 0.7 ]])
#Female adults
conversion_rates_F = np.array([[0.2 , 0.15, 0.1 , 0.05],
                               [0.56, 0.49, 0.42, 0.37],
                               [0.69, 0.6 , 0.53, 0.45],
                               [0.51, 0.45, 0.34, 0.22],
                               [0.23, 0.16, 0.11, 0.05]])


cr_mean = (conversion_rates_S+conversion_rates_M+conversion_rates_F)/3


prices = np.array([[ 5. ,  6. ,  8. ,  8.5],
                   [49. , 53. , 59. , 64. ],
                   [ 8. , 11. , 13. , 17. ],
                   [29. , 36. , 50. , 55. ],
                   [33. , 38. , 45. , 49. ]])

#wine€, wine€€€, limoncello, gin, whisky
costs = np.array([3.5, 40, 6.5, 22, 27])
margin = np.zeros((5, 4))

for i in range(prices.shape[0]):
  for j in range(prices.shape[1]):
    margin[i][j] = prices[i][j] - costs[i]

alphas_S = np.array([0.06, 0.29, 0.06, 0.21, 0.27, 0.11])
alphas_M = np.array([0.08, 0.06, 0.21, 0.16, 0.18, 0.31])
alphas_F = np.array([0.17, 0.09, 0.21, 0.28, 0.17, 0.08])

alphas_mean = (alphas_S + alphas_M + alphas_F)/3

#daily nb of users either random or min max for each user category

sold_items_S = np.array([2.4, 0.3, 0.1, 0.6, 0.17])      #avg of sold items +1)
sold_items_M = np.array([3.5, 2.7, 0.7, 0.9, 1.3])
sold_items_F = np.array([1.5, 2.1, 1.3, 0.5, 0.1])

sold_items_mean = (sold_items_S+sold_items_M+sold_items_F)/3


graph_proba_S= np.array([[0.0, 0.2, 0.35, 0.87, 0.12],
                         [0.51, 0.0, 0.06, 0.26, 0.45],
                         [0.73, 0.08, 0.0, 0.09, 0.55],
                         [0.13, 0.2 , 0.26 , 0.0, 0.58],
                         [0.85, 0.73, 0.4, 0.51, 0.0]]).T

graph_proba_M = np.array([[0.0, 0.36, 0.47, 0.29, 0.1 ],
                          [0.68, 0.0, 0.51, 0.74, 0.46],
                          [0.18, 0.63, 0.0, 0.68, 0.29],
                          [0.41, 0.73, 0.25, 0.0, 0.22],
                          [0.25, 0.39, 0.5 , 0.33, 0.0]]).T

graph_proba_F = np.array([[0.0, 0.38, 0.63, 0.73, 0.26],
                          [0.33, 0.0, 0.18, 0.71, 0.5],
                          [0.4, 0.67, 0.0, 0.87, 0.55],
                          [0.36, 0.45 , 0.24, 0.0 , 0.32],
                          [0.25, 0.59, 0.06, 0.51, 0.0]]).T

graph_proba_mean = (graph_proba_S + graph_proba_M + graph_proba_F)/3

l = 0.8

sec_prod = np.array([[1, 4],
                    [0, 3],
                    [4, 1],
                    [1, 2],
                    [0, 1]])