import numpy as np

# products:
# 1) wine€
# 2) wine€€€
# 3) limoncello
# 4) gin
# 5) whisky

#Students
conversion_rates_S = np.array([[0.95, 0.85, 0.78, 0.62],
                               [0.39, 0.25, 0.13, 0.02],
                               [0.77, 0.60, 0.48, 0.37],
                               [0.8 , 0.65, 0.5 , 0.4],
                               [0.63, 0.48, 0.35, 0.23]])

#Male adults
conversion_rates_M = np.array([[0.37, 0.25, 0.12, 0.01],
                               [0.90, 0.74, 0.69, 0.55],
                               [0.69, 0.44, 0.26, 0.10],
                               [0.75, 0.63, 0.51, 0.39],
                               [0.95, 0.83, 0.71, 0.58]])

#Female adults
conversion_rates_F = np.array([[0.45, 0.30, 0.15, 0.01],
                               [0.65, 0.51, 0.39, 0.23],
                               [0.79, 0.63, 0.51, 0.41],
                               [0.51, 0.39, 0.24, 0.12],
                               [0.41, 0.22, 0.11, 0.02]])

cr_mean = (conversion_rates_S+conversion_rates_M+conversion_rates_F)/3

#prices for every product
prices = np.array([[ 4. ,  6. ,  9. ,  13.],
                   [49. , 53. , 59. , 64. ],
                   [ 8. , 12. , 18. , 25. ],
                   [29. , 36. , 50. , 57. ],
                   [33. , 38. , 45. , 49. ]])

#wine€, wine€€€, limoncello, gin, whisky
costs = np.array([3.5, 40, 6, 22, 27])
costs = np.array([[3.5], [40], [6], [22], [27]])

margin = prices - costs
alphas_S = np.array([0.06, 0.29, 0.06, 0.21, 0.27, 0.11])
alphas_M = np.array([0.08, 0.06, 0.21, 0.16, 0.18, 0.31])
alphas_F = np.array([0.17, 0.09, 0.21, 0.28, 0.17, 0.08])

alphas_mean = (alphas_S + alphas_M + alphas_F)/3


min_daily_users_S = 50
max_daily_users_S = 150
min_daily_users_M = 50
max_daily_users_M = 150
min_daily_users_F = 50
max_daily_users_F = 150

sold_items_S = np.array([2.4, 0.3, 0.1, 0.6, 0.2])      #avg of sold items (+1)
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


l = 0.75
lambda_mat2 = np.array([[0., 1, 0., l, 0.], #wine€ -> wine€€€ and gin as sec prods
                     [0., 0., 0., l, 1], #wine€€€ -> whisky - gin
                     [0., 1., 0., 0., l], #limoncello -> wine€€€ - whisky
                     [l, 0., 1., 0., 0.], #gin -> lim, wine€
                     [0., 0., l, 1., 0.]]) #whisky -> gin - lim
