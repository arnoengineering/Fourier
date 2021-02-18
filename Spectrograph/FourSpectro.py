import numpy as np
from scipy.fftpack import fft
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
# from matplotlib import cm
# from scipy import fft
# from matplotlib.pyplot import specgram

import struct
import pyaudio
import sys


class PlotWin(QtWidgets.QMainWindow):
    def __init__(self):
        super(PlotWin, self).__init__()
        # pyqtgraph stuff
        pg.setConfigOptions(antialias=True)
        self.traces = dict()
        self.app = QtWidgets.QApplication(sys.argv)
        self.win = pg.PlotWidget(title='Spectrum Analyzer')
        self.win.setWindowTitle('Spectrum Analyzer')
        self.win.setGeometry(5, 115, 1910, 1070)
        self.timer = QtCore.QTimer()

        # self.spect = self.form_spect()
        self.waveform = self.format_plots('WAVEFORM', (1, 1), (4096, 255))

        # pyaudio stuff
        self.from_file = False
        # self.num_freq =5
        self.max_time = 5  # s
        # todo set so diff object
        # self.total_list = np.zeros(self.CHUNK)  # titles
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = 1024 * 2

        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            output=True,
            frames_per_buffer=self.CHUNK,
        )
        # waveform and spectrum x points
        self.x = np.arange(0, 2 * self.CHUNK, 2)
        self.f = np.linspace(0, self.RATE // 2, self.CHUNK)

    # def form_spect(self):
    #     # bipolar colormap
    #     img = pg.ImageItem()
    #     self.view.addItem(img)  # , row=1, col=1)
    #     pos = np.array([0., 1., 0.5, 0.25, 0.75])
    #     color = np.array([[0, 255, 255, 255], [255, 255, 0, 255], [0, 0, 0, 255],
    #                       (0, 0, 255, 255), (255, 0, 0, 255)], dtype=np.ubyte)
    #     cmap = pg.ColorMap(pos, color)
    #     lut = cmap.getLookupTable(0.0, 1.0, 256)
    #
    #     # set colormap
    #     img.setLookupTable(lut)
    #     img.setLevels([-50, 40])
    #     return img

    def format_plots(self, plot, plt_num, range_data, log_x=False, log_y=False):
        def form_dat(da, logg, ori='bottom'):
            t_ls = [0, da // 2, da] if not logg else [1, np.log10(da) // 2, np.log10(da)]
            t_range = [(d2, str(d2)) for d2 in t_ls]
            t_ax = pg.AxisItem(orientation=ori)
            t_ax.setTicks([t_range])
            return t_ax

        x_lab = form_dat(range_data[0], log_x, 'bottom')
        y_lab = form_dat(range_data[1], log_y, 'left')
        return self.win.addPlot(
                title=plot, row=plt_num[0], col=plt_num[1], axisItems={'bottom': x_lab, 'left': y_lab})

    def set_plotdata(self, name, data_x, data_y):
        if name in self.traces:
            self.traces[name].setData(data_x, data_y)
        else:
            if name == 'waveform':
                self.traces[name] = self.waveform.plot(pen='c', width=3)
                self.waveform.setYRange(0, 255, padding=0)
                self.waveform.setXRange(0, 2 * self.CHUNK, padding=0.005)
            # if name == 'spectrum':
            #     self.traces[name] = self.spectrum.plot(pen='m', width=3)
            #     self.spectrum.setLogMode(x=True, y=True)
            #     self.spectrum.setYRange(-4, 0, padding=0)
            #     self.spectrum.setXRange(
            #         np.log10(20), np.log10(self.RATE / 2), padding=0.005)

    def save_data(self):
        pass

    def update(self):

        aud_im = 1
        while aud_im:  # while speaking
            if self.from_file:
                data_np = self.from_file  # placeholder
            else:
                data = self.stream.read(self.CHUNK)
                data_int = struct.unpack(str(2 * self.CHUNK) + 'B', data)
                data_np = np.array(data_int, dtype='b')[::2] + 128
                self.set_plotdata(name='waveform', data_x=self.x, data_y=data_np, )
            # fft
            freq_fft = np.log10(np.abs(fft(data_np)))
            print(freq_fft.min(), freq_fft.max())
            # self.total_list = np.vstack((self.total_list, freq_fft))
            # todo take lastx

            # # print(self.total_list.shape)
            # if self.total_list.shape[0] > 500:  # max size, add so is funct of max y_time
            #     self.total_list = self.total_list[1:, :]  # get rid of first
                # break
            # plot_img(total_list)
            print(data_np)

            # self.set_plotdata(name='spectrum', data_x=self.f, data_y=freq_fft)
            # self.spect.setImage(self.total_list)
            # self.set_plotdata(name='spect', data_x=self.f, data_y=freq_fft)

            # plot_img(total_list)
        self.save_data()

    def animation(self):
        self.timer.setInterval(20)
        self.timer.timeout.connect(self.update)
        self.timer.start()
        self.start()


audio_app = PlotWin()
audio_app.animation()
sys.exit(audio_app.exec_())
