# import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from Four3b1b.FourierImage import *


def compon_ls(ls):
    real_ls = [j.real for j in ls]
    imag_ls = [j.imag for j in ls]
    return real_ls, imag_ls


def ani_circle(dt):  # assume each has st_pos, pos
    # dt coming in np.linsp(2pi)
    # bring dt
    # vector p
    v_or = np.array([0, 0], 'float64')  # change from or to last vec
    for v in vector[:5]:  # colomns
        # print(v)
        # print(v[1])
        # freq from x, maybe add omega
        ang_vel = v[0].real / (2 * np.pi)
        # pos relative to vect_origin
        v_pos = np.array([v[1].real, v[1].imag])  # cartesian

        # print(v_vec)
        v_cros = np.cross([v[1].real, v[1].imag, 0], [0, 0, ang_vel]) * dt
        v_pos += v_cros[:1]
        # print(v_pos)
        vec_arr = np.random.random((2, 50))
        # for r in range(2):
        #     lin_vec = np.linspace(v_or[r], v_pos[r])
        #     print(vec_arr)
        #     print(lin_vec)
        #     vec_arr[r] = lin_vec
            # hope array from(x0,y0)->(x,y)
        # print(vec_arr)
        plt.plot([v_or[0], v_pos[0]], [v_or[1], v_pos[1]])  # plot if vpos is in x,y np array

        plot_circ(np.abs(v[1]), v_or)
        v_or += v_pos  # adds vectors tohetler
        print(v_or)

    plt.scatter(v_or[0], v_or[1], c='r')  # scince or updates at last vect, the tip, use color


def plot_circ(rad, cent):
    # either plt.circ or scatter
    theta = np.linspace(0, 2 * np.pi, 100)  # range for plot
    cir_x = cent[0] + np.cos(theta) * rad
    cir_y = cent[1] + np.sin(theta) * rad
    plt.plot(cir_x, cir_y)


vector = four_dict.T  # transpose
# print(vector)
# show image in background
# anim circle
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ani_circle(0)
# plt.anim = FuncAnimation(fig, ani_circle)
plt.show()
