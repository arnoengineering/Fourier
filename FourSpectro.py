from PIL import Image
from pydub import AudioSegment
from pydub.generators import Sine
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import struct
import pyaudio
# get sound rom im
pic_file = ''
sound_file = ''
return_img = ''
ret_sound = ''

img = Image.open(pic_file)
img.convert('LA')
width, height = img.size
ratio = height / width


x_size = 500
y_size = int(500 * ratio)
pix_map = img.load()


# sound values
audio = AudioSegment.silent(duration=0)
duration_per_pix = 100  # ms per line
frequncy = []  # logscale mage image 500, 500...

# logscale x, do frequncy, ave x


def gen_sound(hz, pix_val):
    # scale amp, freq
    am_scale = 500
    freq = frequncy[hz]  # index, fr is logscale 500
    s_wav = Sine(freq)
    wav = s_wav.to_audio_segment(duration_per_pix, am_scale*pix_val)
    wav = wav.fade_in(0.1 * duration_per_pix).fade_out(0.1 * duration_per_pix)
    return wav


for y in range(y_size):
    row_seg = AudioSegment.silent(duration=0)
    for x in range(x_size):
        pix = pix_map.getpixel((x,y))
        seg = gen_sound(x, pix)
        row_seg.overlay(seg)
    audio += row_seg


def audio_to_image():
    aud_im = 1
    img_obj = 3
    y = 0
    while aud_im:  # while speaking
        # fft
        freq_fft = {}  # f, amp
        # wait time per line, ave all fft
        y += 1
        line = []
        if y > 500:  # max size
            break
        for f, amp in freq_fft.items():
            # see closes f in freq
            ind = frequncy.index(f)  # closest
            img_obj.pix(ind, y) = amp
        print(line)
    img_obj.save()
    # write to pic_file





def process_image():
    """turn spec image into real image"""