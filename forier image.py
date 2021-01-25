import numpy as np
from scipy.fft import fft, fftfreq
from svgpathtools import svg2paths
# from matplotlib import pyplot as plt


file = 'Do_Mayor_armadura.svg'
poly_file = 'media/poly_file.csv'
n_file = 'polynomial_form.txt'

# get path from svg, # todo check numper of paths, sub-paths
paths, attributes = svg2paths(file)

# number of datapoints also number of f outputs
input_num = 200  # maybe ad loop of num, range
input_list = np.linspace(0, 1, input_num)
print(input_list)


poly_ls = []  # exdend or add list
for path in paths:
    for sub_path in path:
        poly_path = sub_path.poly()
        # this is a polynomial_object
        poly_ls.extend(poly_path)

# since this is list, back to  poly
polynom = np.poly1d(poly_ls)

# get fourier transform
for_ls = fft(polynom(input_list))
four_size = for_ls.size()
four_freq = fftfreq(four_size, 1 / four_size)

# get largest indexes, maybe else just del


def save_coef():
    with open(poly_file) as f:
        coefs = f.readlines()

    poly_ord = len(coefs)

    newlines = []
    for n in range(poly_ord):
        li = coefs[n]
        x_val = poly_ord - (n + 1)
        if x_val > 0:
            st = f'({li})x^({x_val}) +'
        else:
            st = f'({li})'
        newlines.append(st)

    with open(n_file) as f:
        f.writelines(newlines)


def graph_four(freq, amp):
    # amp--or initval:
    # a+bj
    t = range(10)  # place
    val = amp * np.exp(freq*t*1j)
    # ???? initial rad = abs(amp), inital angle theta = amp.imag/amp.real
    # plot val as circle radius amp a


print('\nfor in\n')
# print(for_in)

# np.savetxt(poly_file, polynom.coeffs)
