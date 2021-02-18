
# from PIL import Image

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
# todo save both
# self.view = self.win.addViewBox()

        # self.plot = pg.PlotItem()
        # self.win.addPlot(self.plot, row=1, col=1)

class StremObj:
    def __init__(self)
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = 1024 * 2
        self.total_list = np.zeros(self.CHUNK)  # titles

        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            output=True,
            frames_per_buffer=self.CHUNK,
        )

def stream_obj(form):
#     st = p.open(format=form, channels=Channels, rate=rate,
#                 output=True, input=True, frames_per_buffer=Chuck)
#     return st

def log_bin(hz_ls, hz_vals):  # hz_list be dict?
# cor_freq = []
#     dig_index = np.digitize(hz_ls, frequency)
#     for num, n_val in enumerate(dig_index):
#         cor_freq[n_val] += hz_vals[n_val]
#     return cor_freq
#
#
# def gen_sound(hz, pix_val):
#     # scale amp, freq
#     freq = frequency[hz]
#     samples = (np.sin(2*np.pi*np.arange(rate*duration_per_pix)*freq/rate)).astype(np.float32).tobytes()
#     sine_wav = pix_val * np.sin(samples)  # get value per x
#     return sine_wav
## stream_out = stream_obj(pyaudio.paFloat32)
# stream = stream_obj(Format)
#
# x_size = 500
#
# # sound values
# duration_per_pix = 0.1  # ms per line
# frequency = np.geomspace(20, 20000, x_size)
#
# # plots
# x_data = np.linspace(0, 2 * Chuck, Chuck)
# fig, ax = plt.subplots(2, 2)
# line = set_f_plots((0, 0))
# line2 = set_f_plots((1, 0))
# num_freq = 5
#
# freq_lines = [set_f_plots((1, 1), col=c_map(c_val), log=False) for c_val in range(num_freq)]
#
# plt.show(block=False)
#
# from_where = 'sound'  # input('type')
# if from_where == 'im':
#     # get sound rom im
#     pic_file = 'media/DSC_0178.JPG'
#     aud_from_im()
#
# elif from_where == 'sound':
#     audio_to_image()