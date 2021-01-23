import numpy as np
from scipy.fft import fft, fftfreq
from svgpathtools import svg2paths  # Path, Line, QuadraticBezier, CubicBezier, Arc,
# from matplotlib import pyplot as plt
"""BLUR,CONTOUR
DETAIL
EDGE_ENHANCE
"""

"""smooth image
convert to svg
get array from image,
 np.exp^360"""

file = 'Do_Mayor_armadura.svg'
poly_file = 'poly_file.csv'

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


"""
svg:
extend all paths 
for path in paths:
decode path
p_total.append[path]

convert svg?
get img svg y
get curves y
for each curve get fft y
comb curves y

for each freq disp, show time varimg abs
pos, vs time
show pos variang
let circles run with initial f
do plot circ
plot sum f
save animation

From: https://github.com/skyzh/fourier-transform-drawing/blob/master/analysis/analysis.py
fp = open("fft_data.json")
fft_data = json.load(fp)
complex_real = array(list(map(lambda d: d["x"], fft_data)))
complex_imag = array(list(map(lambda d: d["y"], fft_data)))
complex_data = complex_real + 1j * complex_imag

# print(complex_data)

n = len(complex_data)
t = linspace(0, 1, n)

# print(t)

freq_range = 150

freq_map = dict()

for k in arange(-freq_range, freq_range + 1):
    xx = -2 * pi * k * 1j * t
    c_n = sum(complex_data * exp(xx)) / n
    # print(k, c_n)
    freq_map[int(k)] = { 'x': real(c_n), 'y': imag(c_n) }

print(json.dumps(freq_map))
"""
