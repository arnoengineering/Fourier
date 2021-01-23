import numpy as np
from matplotlib import pyplot as plt
# from matplotlib.cm import get_cmap


def compon_ls(ls):
    real_ls = [j.real for j in ls]
    imag_ls = [j.imag for j in ls]
    return real_ls, imag_ls


rad_vals = np.pi * np.linspace(0, 2, 10)
sequence = np.arange(10)
initial_vals = np.array([np.exp(1j*x) for x in rad_vals])
plots = 3
fig, axs = plt.subplots(plots, 2)
frequency = [np.exp(2*np.pi*1j*f) for f in range(plots)]

print(initial_vals)
for n in range(plots):
    # c is color
    # if map not correct pass range c=cm['cm_name'](range(len(inval)))
    in_real, in_im = compon_ls(initial_vals)
    print(f'\nfr_n{frequency[n]}')
    fr_vals = frequency[n] * initial_vals
    print('\n fr_vals')
    print(fr_vals)
    fr_real, fr_imag = compon_ls(fr_vals)

    axs[n, 0].scatter(in_real, in_im, c=sequence) #, c=get_cmap('gist_rainbow'))
    axs[n, 1].scatter(fr_real, fr_imag, c=sequence) #, c=get_cmap('gist_rainbow'))

plt.show()
