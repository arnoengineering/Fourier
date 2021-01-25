import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import fft
# from scipy.signal import spectrogram
import struct
import pyaudio


# get sound rom im
pic_file = 'media/DSC_0178.JPG'
sound_file = ''
return_img = ''
ret_sound = ''

Chuck = 1024 * 2
Format = pyaudio.paInt16
Channels = 1
rate = 44100

p = pyaudio.PyAudio()


def stream_obj(form):
    st = p.open(format=form, channels=Channels, rate=rate,
                output=True, input=True, frames_per_buffer=Chuck)
    return st


# axis for local plot
def set_f_plots(plot_num, col='b', log=True):
    # also add for num line in p
    # x_val, freq
    if log:
        x_val = frequency
    else:
        x_val = x_data
    line_val, = ax[plot_num[0], plot_num[1]].plot(x_val, np.random.rand(len(x_val)), c=col)
    ax[plot_num].set_xlim(0, Chuck)
    ax[plot_num].set_ylim(0, 255)
    return line_val


stream_out = stream_obj(pyaudio.paFloat32)
stream = stream_obj(Format)


img = Image.open(pic_file)
img.convert('LA')
width, height = img.size
ratio = height / width

x_size = 500
y_size = int(500 * ratio)
pix_map = img.load()

# sound values
duration_per_pix = 0.1  # ms per line
frequency = np.geomspace(20, 20000, x_size)

# plots
x_data = np.linspace(0, 2 * Chuck, Chuck)
fig, ax = plt.subplots(2, 2)
line = set_f_plots((0, 0), log=False)
line2 = set_f_plots((1, 0))

freq_lines = [set_f_plots((1, 1), col=c_val, log=False) for c_val in ['b', 'r', 'g', 'p', 'v']]

plt.show(block=False)


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
#
#
# def plot_img(lines):  # create 3d image
#     plt.pcolormesh(lines)
#     # plt.contourf(lines)
#     plt.xticks(frequency)
#     plt.xlabel('Frequency')
#     plt.yticks([duration_per_pix * n for n in range(lines.shape[0])])
#     plt.ylabel('Time')


def audio_to_image(from_file=False):
    # total_list = np.array(frequency)  # titles
    aud_im = 1
    while aud_im:  # while speaking
        if from_file:
            data_np = []  # placeholder
        else:
            data = stream.read(Chuck)
            data_int = struct.unpack(str(2 * Chuck) + 'B', data)
            data_np = np.array(data_int, dtype='b')[::2] + 128

        # fft
        freq_fft = fft.fft(data_np)
        fft_freq = fft.fftfreq(freq_fft.size, 1 / rate)  # todo merge for argsort
        max_freq_vals = np.argpartition(freq_fft, -5)[-5:]
        max_freq = fft_freq[max_freq_vals]
        # for n, f in enumerate(max_freq):
        #     fr = np.sin(np.linspace(0, 2 * np.pi, 500))
        #     freq_lines[n].set_ydata(fr)

        line.set_ydata(data_np)
        line2.set_ydata(freq_fft)  # live color x

        fig.canvas.draw()
        fig.canvas.flush_events()


def aud_from_im():
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


audio_to_image()
