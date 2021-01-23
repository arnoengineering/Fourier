import numpy as np

x_ls = range(4)
poly_nom = np.poly1d([1, 2, 3])
print(poly_nom(x_ls[0]))
print(poly_nom(x_ls))
