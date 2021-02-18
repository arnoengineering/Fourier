import numpy as np
from numpy.polynomial import polynomial as polyn
import os
# import pandas as pd
from scipy.fft import fft, fftfreq
from svgpathtools import svg2paths  # , # wsvg, path

from matplotlib import pyplot as plt


path2 = os.path.dirname(__file__)
max_size = 10
os.chdir(path2)
file = 'Do_Mayor_armadura.svg'
poly_file = '../media/poly_file.csv'
n_file = 'scale.svg'


# paths = parse_path()
paths, attributes = svg2paths(file)

# number of datapoints also number of f outputs
input_num = 200  # maybe ad loop of num, range
input_list = np.linspace(0, 1, input_num)

poly_ls = polyn.Polynomial([0])  # exdend or add list

for s_path in paths:
    for sub_path in s_path:
        poly_path = sub_path.poly()
        # this is a polynomial_object

        temp_poly = polyn.Polynomial(poly_path.coeffs)
        plt.scatter(temp_poly.real, poly_vals.imag)
        poly_ls += temp_poly
        break


# since this is list, back to  poly
# polynom = poly_ls

# get fourier transform
print(poly_ls)
poly_vals = poly_ls(input_list)
print(poly_vals)
plt.scatter(poly_vals.real, poly_vals.imag)
plt.show()

for_ls = fft(poly_vals)
four_size = for_ls.size
four_freq = fftfreq(four_size, 1 / four_size).real

four_mag = sum(np.abs(for_ls)) / 50

four_norm = for_ls  # / four_mag
four_dict = np.vstack((four_freq, four_norm))  # frequncy, amp


print('\nfor in\n')
# print(for_in)

# np.savetxt(poly_file, polynom.coeffs)
