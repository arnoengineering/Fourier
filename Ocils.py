import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm

import struct
import pyaudio

"""
Input Device id  0  -  Microsoft Sound Mapper - Input
Input Device id  1  -  Analogue 1 + 2 (Focusrite Usb A
Input Device id  2  -  Microphone (HD Pro Webcam C920)
Input Device id  3  -  Microphone (CORSAIR ST100 Heads
Input Device id  4  -  Line (Microsoft)
"""

sound_file = ''
return_img = ''
ret_sound = ''

# sound init
scale = 255 / 2
Chuck = 1024 * 2
Format = pyaudio.paInt16
Channels = 1
rate = 44100
c_map = cm.get_cmap('Set1')

p = pyaudio.PyAudio()


def stream_obj(form, ch=Chuck, **kwargs):
    st = p.open(format=form, channels=Channels, rate=rate,
                output=True, input=True, frames_per_buffer=ch, **kwargs)
    return st


def set_f_plots(plot_num, col='b', y_lim=255, x_lim=Chuck):
    # also add for num line in p
    # x_val, freq
    line_val, = ax[plot_num[1]].plot(x_data, np.random.rand(len(x_data)), c=col)

    ax[plot_num[1]].set_xlim(0, x_lim)
    ax[plot_num[1]].set_ylim(0, y_lim)
    return line_val


def audio_to_image():
    aud_im = 1
    while aud_im:  # while speaking
        tot_dat = np.zeros((2, Chuck))
        for n, stream in enumerate(streams.keys()):
            data = stream.read(Chuck)
            data_int = struct.unpack(str(2 * Chuck) + 'B', data)
            data_np = np.array(data_int, dtype='b')[::2] + 128
            tot_dat[n] = data_np

            streams[stream].set_ydata(data_np)  # sets line data for both

        print(tot_dat)
        total_line.set_data(tot_dat[0], tot_dat[1])  # get np returned then get valurs for eack where x,y = l1,s2
        fig.canvas.draw()
        fig.canvas.flush_events()


stream_out = stream_obj(pyaudio.paFloat32)
stream_r = stream_obj(Format, input_device_index=2)
stream_l = stream_obj(Format, input_device_index=1)  # add input

x_size = 500

# sound values
duration_per_pix = 0.1  # ms per line
frequency = np.geomspace(20, 20000, x_size)

# plots
# in 1 x/t second y/t third y/x
x_data = np.linspace(0, 2 * Chuck, Chuck)
fig, ax = plt.subplots(1, 3)


streams = {stream_l: set_f_plots((0, 0)), stream_r: set_f_plots((0, 1))}  # l = x, stream: line of stream

total_line, = ax[2].plot(x_data, np.random.rand(len(x_data)), c='r')
ax[2].set_xlim(0, 255)
ax[2].set_ylim(0, 255)

plt.show(block=False)

audio_to_image()
