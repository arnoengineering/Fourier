import numpy as np
from PIL import Image


class ImageGen:
    def __init__(self, img_f, freq_ax=0, freq_min=20, freq_max=20000, harmonics=0,
                 duration_pix=0.1, sample_rate=44100, log_freq=True, freq_cnt=500):
        self.img = Image.open(img_f)
        self.img_data = self.form_img()

        self.axis = freq_ax
        self.log_freq = log_freq
        self.freq_num = freq_cnt
        self.harmonics = harmonics
        self.sample_rate = sample_rate
        self.harm_range = 40  # hz

        self.duration_pix = duration_pix
        if self.log_freq:
            self.freq_range = np.logspace(freq_min, freq_max, self.freq_num)
        else:
            self.freq_range = np.linspace(freq_min, freq_max, self.freq_num)
        self.sound_arr = np.array([])

    def loop_img(self):
        for c in self.img_data.columns:
            self.sound_col(c)

    def form_img(self):
        self.img.convert('LA')
        width, height = self.img.size
        ratio =  width / height if self.axis else height / width

        size = int(self.freq_num * ratio)
        self.img.resize(size)
        pix_map = self.img.load()
        return pix_map

    def sound_col(self, col):
        # if x rem trans
        co_wave = np.array([])  # todo fix
        for num, pix in enumerate(col):
            co_wave += self.create_sound(num, pix)
        np.concatenate(self.sound_arr, co_wave)

    def create_sound(self, num, pix):
        pix_wave = []
        base_freq = self.freq_range[num]
        for n in range(self.harmonics):
            freq = base_freq + n * self.harm_range
            pix_wave += pix * np.sin(freq * 2 * np.pi * np.linspace(0, self.duration_pix,
                                                                    int(self.duration_pix * self.sample_rate)))

        return pix_wave
