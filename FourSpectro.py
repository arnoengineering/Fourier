import numpy as np
from scipy import fft
from scipy.signal import spectrogram

import matplotlib.pyplot as plt
from matplotlib import cm

import struct
import pyaudio

from PIL import Image

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


def stream_obj(form):
    st = p.open(format=form, channels=Channels, rate=rate,
                output=True, input=True, frames_per_buffer=Chuck)
    return st


# axis for local plot
def set_f_plots(plot_num, col='b', log=False):
    # also add for num line in p
    # x_val, freq
    if log:
        x_val = frequency
    else:
        x_val = x_data
    line_val, = ax[plot_num[0], plot_num[1]].plot(x_val, np.random.rand(len(x_val)), c=col)

    ax[plot_num[0], plot_num[1]].set_xlim(0, len(x_val))
    ax[plot_num[0], plot_num[1]].set_ylim(0, 255)
    return line_val


def log_bin(hz_ls, hz_vals):  # hz_list be dict?
    cor_freq = []
    dig_index = np.digitize(hz_ls, frequency)
    for num, n_val in enumerate(dig_index):
        cor_freq[n_val] += hz_vals[n_val]
    return cor_freq


def gen_sound(hz, pix_val):
    # scale amp, freq
    freq = frequency[hz]
    samples = (np.sin(2*np.pi*np.arange(rate*duration_per_pix)*freq/rate)).astype(np.float32).tobytes()
    sine_wav = pix_val * np.sin(samples)  # get value per x
    return sine_wav


def plot_img(lines):  # create 3d image

    plt.pcolormesh(lines)
    # plt.contourf(lines)
    plt.xticks(frequency)
    plt.xlabel('Frequency')
    plt.yticks([duration_per_pix * n for n in range(lines.shape[0])])
    plt.ylabel('Time')


def save_spec(hz_data):
    # list split
    spectrogram(hz_data)
    # plt.colorbar()
    plt.imshow()


def audio_to_image(from_file=None):
    total_list = np.zeros(Chuck)  # titles
    aud_im = 1
    while aud_im:  # while speaking
        if from_file is not None:
            data_np = from_file  # placeholder
        else:
            data = stream.read(Chuck)
            data_int = struct.unpack(str(2 * Chuck) + 'B', data)
            data_np = np.array(data_int, dtype='b')[::2] + 128

        # fft
        freq_fft = fft.fft(data_np)
        # get frequency
        fft_freq = fft.fftfreq(freq_fft.size, 1 / rate)  # todo merge for argsort
        max_freq_vals = np.argpartition(freq_fft, -num_freq)[-num_freq:]  # gets indexes of max vals
        max_freq = fft_freq[max_freq_vals]

        # find values
        max_freq_ab = np.abs(freq_fft[max_freq_vals])
        tot_scale = np.sum(max_freq_ab)

        for n, f in enumerate(max_freq):
            print(f)
            # scale is vaule of f in freq _fft / sum* scale, logscale
            f_scale = tot_scale
            fr = scale + scale * f_scale * np.sin(np.log10(f) * np.linspace(0, 20, Chuck))
            freq_lines[n].set_ydata(fr)

        total_list = np.vstack((total_list, data_np))
        #
        print(total_list.shape)
        if total_list.shape[0] > 100:  # max size
            break
        # plot_img(total_list)

        line.set_ydata(data_np)
        line2.set_ydata(np.abs(freq_fft))  # live color x

        fig.canvas.draw()
        fig.canvas.flush_events()
    save_spec(total_list)


def aud_from_im():
    img = Image.open(pic_file)
    img.convert('LA')
    width, height = img.size
    ratio = height / width

    y_size = int(500 * ratio)
    pix_map = img.load()

    row_seg = []
    for y in range(y_size):
        seg = np.linspace(0, np.pi * 2, int(rate * duration_per_pix))
        for x in range(x_size):
            pix = pix_map.getpixel((x, y))
            seg += gen_sound(x, pix)
            # for sum, plot x freq*1
        row_seg.extend(seg)
    stream_out.write(row_seg)
    stream_out.stop_stream()
    stream_out.close()
    audio_to_image(from_file=True)


stream_out = stream_obj(pyaudio.paFloat32)
stream = stream_obj(Format)

x_size = 500

# sound values
duration_per_pix = 0.1  # ms per line
frequency = np.geomspace(20, 20000, x_size)

# plots
x_data = np.linspace(0, 2 * Chuck, Chuck)
fig, ax = plt.subplots(2, 2)
line = set_f_plots((0, 0))
line2 = set_f_plots((1, 0))
num_freq = 5

freq_lines = [set_f_plots((1, 1), col=c_map(c_val), log=False) for c_val in range(num_freq)]

plt.show(block=False)

from_where = 'sound' # input('type')
if from_where == 'im':
    # get sound rom im
    pic_file = 'media/DSC_0178.JPG'
    aud_from_im()

elif from_where == 'sound':
    audio_to_image()

