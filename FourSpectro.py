import numpy as np
from PIL import Image
from pydub import AudioSegment

import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.signal import spectrogram
import struct
import pyaudio
# get sound rom im
pic_file = ''
sound_file = ''
return_img = ''
ret_sound = ''

Chuck = 1024 * 2
Format = pyaudio.paInt16
Channels = 1
rate = 44100

p = pyaudio.PyAudio()
stream = p.open(
                format=Format,
                chanels=Channels,
                rate=rate,
                output=True,
                input=True,
                frames_per_buffer=Chuck)

stream_out = p.open(
                format=pyaudio.paFloat32,
                chanels=Channels,
                rate=rate,
                output=True,
                input=True,
                frames_per_buffer=Chuck)


# axis for local plot
fig, (ax, ax2, ax_sine, ax_his) = plt.subplots(4)
x_data = np.arrange(0, 2 * Chuck, 2)
line, = ax.plot(x_data, np.random.rand(Chuck))
ax.set_xlim(0, Chuck)
ax.set_ylim(0, 255)

f_data = np.linspace()
line2, = ax2.plot(f_data, np.random.rand(Chuck))
ax2.set_xlim(0, Chuck)
ax2.set_ylim(0, 255)

img = Image.open(pic_file)
img.convert('LA')
width, height = img.size
ratio = height / width

x_size = 500
y_size = int(500 * ratio)
pix_map = img.load()


# # sound values
# audio = AudioSegment.silent(duration=0)
duration_per_pix = 0.1  # ms per line
frequency = np.logspace(20, 20000, x_size)


def max_freq(hz_ls, num):
    high_freq = {}
    c_l = hz_ls
    for n in range(num):
        max_fr = max(c_l)
        high_freq[frequency[c_l.index(max_fr)]] = max_fr
        c_l.remove(max_fr)
    return high_freq


def gen_sound(hz, pix_val):
    # scale amp, freq
    freq = frequency[hz] 
    samples = (np.sin(2*np.pi*np.arange(rate*duration_per_pix)*freq/rate)).astype(np.float32).tobytes()
    sine_wav = pix_val * np.sin(samples)  # get value per x
    # s_wav = freq  # sine(freq)
    # wav = s_wav.to_audio_segment(duration_per_pix, am_scale*pix_val)
    # wav = wav.fade_in(0.1 * duration_per_pix).fade_out(0.1 * duration_per_pix)
    return sine_wav


def plot_img(lines):  # create 3d image
    plt.pcolormesh(lines)
    # plt.contourf(lines)
    plt.xticks(frequency)
    plt.xlabel('Frequency')
    plt.yticks([duration_per_pix * n for n in range(lines.shape[0])])
    plt.ylabel('Time')

    pass


def audio_to_image(from_file=False):
    total_list = np.array(frequency)  # titles
    aud_im = 1
    while aud_im:  # while speaking
        if from_file:
            data_int = []  # placeholder
        else:
            data = stream.read()
            data_int = struct.unpack(str(2 * Chuck) + 'B', data)
        line.set_ydata(data_int)
        fig.canvas.draw()
        fig.canvas.flush_events()

        # fft
        freq_fft = fft(data_int)
        fft_freq = fft.fftfreq(freq_fft)
        line2.set_ydata(freq_fft)  # live color x
        # gen tone for each sine
        freq_ls = max_freq(freq_fft, 5)
        # sum_x_fr = sum(freq_ls)
        for fr in freq_ls:
            line_fr = 2 * np.pi * np.arrage(rate*duration_per_pix) * fr/rate
            ax_sine.plot(line_fr, np.sin(line_fr))

        total_list = np.vstack(total_list, data_int)

        if total_list.shape[0] > 500:  # max size
            break
        plot_img(total_list)

    spec = spectrogram(total_list)
    # save spec
    plt.show(spec)
    img_obj = Image.fromarray(total_list)
    # write to pic_file
    process_image(img_obj)


def process_image(img_o):
    """turn spec image into real image"""
    # save
    img_o.save(return_img)


row_seg = []
for y in range(y_size):
    # AudioSegment.silent(duration=0)
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

"""use spect to view, maybe just save, 
use yt for live waveform, save data to np array for spextrograme at end
https://www.youtube.com/watch?v=TJGlxdW7Fb4&ab_channel=SteveBrunton
https://www.youtube.com/watch?v=AShHJdSIxkY&t=1s&ab_channel=MarkJay
get audio evey sample, save raw data to np array, then do fft ard rest yt
show image of pic in
# def test_pos(val):
#     i = 0
#     for f in frequency:
#         if val > f:
#             break
#         i += 1
#     return frequency[i - 1]
*1 for feq
seg += get_freq: get s_wave
then for duration/pixel
seg_tot[duration_pix: duration_pix + chuck] = seg_tot
 after all read from file, then plot,... same as if from live except stream

do
get image, yes
loop though image: yes
get sinedata
save sinedata
save to file, then
plot live below

while mike yes
get audio: yes
decode yes
fft 
save to array yes
plot decode and live, 
plot 5 largest fequcy on dif subplot, diff colors
do spect, yes
save spect
"""