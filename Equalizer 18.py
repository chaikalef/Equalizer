
# coding: utf-8

# In[1]:


import numpy as np
from math import (sin, pi)
import wave
import sys
import pygame
from time import sleep,time

from PyQt5.QtWidgets import (QDialog, QApplication, QWidget,
                             QVBoxLayout, QHBoxLayout,
                             QDesktopWidget, QFileDialog,
                             QSlider, QPushButton, QLabel, 
                             QCheckBox, QLCDNumber)
from PyQt5.QtCore import (Qt, QRunnable, pyqtSlot, QThreadPool)
from PyQt5.QtGui import QPixmap

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

        
        
class Worker(QRunnable):
    def __init__(self, function, *args, **kwargs):
        super(Worker, self).__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs
        

    @pyqtSlot()
    def run(self):
        self.function(*self.args, **self.kwargs)



class Main_Window_class(QDialog):
    def __init__(self):
        super().__init__()
        self.initUI()

        
    def initUI(self):
        ### Hyperparameters ###
        self.nlabels = 10
        
        self.checkboxes_lables = ['Клиппинг', 'Энвелоп']
        self.btns_lables = ['Воспроизвести', 'Пауза', 'Остановить']
        self.app_name = 'Эквалайзер'
        
        self.sld_min = -50
        self.sld_max = 50
        self.sld_def = 0
        self.sld_interval = 10
        self.sld_step = 1
        #######################
        
        ### Global variables ###
        self.path_to_pull = None
        
        self.nchannels = None        # number of channels
        self.sampwidth = None        # number of bytes per sample
        self.framerate = None        # number of frames per second
        self.nframes = None          # total number of frames
        self.comptype = None         # compression type 
        self.compname = None         # compression type name
        self.elem_per_herz = None
        self.koeff = 1000            # коэффициент прореживания
        self.types = {}
        self.buffer_size = None
        self.buffer_cnt = 0
        
        self.music_is_playing = False
        self.threadpool = QThreadPool()
        print(self.threadpool.maxThreadCount())

        self.music_worker = None
        self.sld1_worker = None
        self.sld2_worker = None
        self.sld3_worker = None
        self.sld4_worker = None
        self.sld5_worker = None
        self.sld6_worker = None
        self.sld7_worker = None
        self.sld8_worker = None
        self.sld9_worker = None
        self.sld10_worker = None
        self.checkbox1_worker = None
        self.checkbox2_worker = None
        
        self.min_freq = 0
        self.max_freq = None
        
        self.channels = []
        self.spectrum = None
        self.spectrum_original = None
        self.spectrum_kliping = None
        self.spectrum_envelop = None
        self.channels_original = []
        self.channels_kliping = []
        self.channels_envelop = []
        
        self.bands = [[], []]
        self.labels = []
        ########################
        
        ### Links ###
#         https://habrahabr.ru/post/113239/
#         http://old.pynsk.ru/posts/2015/Nov/09/matematika-v-python-preobrazovanie-fure/
#         http://www.7not.ru/articles/klip_cool.phtml
#         https://martinfitzpatrick.name/article/multithreading-pyqt-applications-with-qthreadpool/
#         https://www.tutorialspoint.com/pyqt/pyqt_qlineedit_widget.htm
#         https://stackoverflow.com/questions/21887862/python-chorus-effect-and-meaning-of-audio-data
#         http://websound.ru/articles/sound-processing/effects.htm
#         http://www.auditionrich.com/lessons/kak-v-adobe-audition-ispolzovat-effekt-envelope-follower.html
#         https://works.doklad.ru/view/M_cHUqES_HE/2.html
        #############
        
        self.path_to_pull = QFileDialog.getOpenFileName(self, 'Выберите .wav файл')[0]
        
        self.pull_music()
        self.create_bands()
        self.create_lables()
        self.create_LCD_numbers()
        self.create_sliders()
        self.create_checkboxes()
        self.create_buttons()
        self.create_graphics()
        self.create_interface()
    
    
    def pull_music(self):
        wav = wave.open(self.path_to_pull, mode = 'r')
        self.types = {
            1: np.int8,
            2: np.int16,
            4: np.int32
        }

        (self.nchannels, self.sampwidth,
         self.framerate, self.nframes,
         self.comptype, self.compname) = wav.getparams()
        
        self.max_freq = self.framerate // 2
        self.buffer_size = self.framerate

        content = wav.readframes(self.nframes)
        samples = np.fromstring(content, dtype = self.types[self.sampwidth])

        for i in range(self.nchannels):
            self.channels.append(samples[i::self.nchannels])
        self.channels = np.array(self.channels)
        
        self.channels_original = self.channels.copy()
        
        self.checkbox1_worker = Worker(self.doing_kliping, self.channels)
        self.threadpool.start(self.checkbox1_worker)
        
        self.checkbox2_worker = Worker(self.doing_envelop, self.channels)
        self.threadpool.start(self.checkbox2_worker)
        
        self.spectrum = np.fft.rfft(self.channels_original)
        self.spectrum_original = self.spectrum.copy()
        
        pygame.mixer.pre_init(frequency = self.framerate,
                              size = -self.sampwidth * 8,
                              channels = self.nchannels)
        pygame.init()
        
        
    def create_bands(self):
        step = (self.max_freq - self.min_freq) // 2**self.nlabels
        
        self.bands[0].append(self.min_freq)
        self.bands[1].append(self.min_freq + step)
        
        for i in range(1, self.nlabels - 1):
            self.bands[0].append(self.bands[1][i - 1])
            self.bands[1].append(self.bands[0][i] + 2**i * step)
        
        self.bands[0].append(self.bands[1][self.nlabels - 2])
        self.bands[1].append(self.max_freq)

        for i in range(self.nlabels):
            self.labels.append(str(self.bands[0][i]) + ' - ' + str(self.bands[1][i]))
    
    
    def create_lables(self):
        self.label_1 = QLabel(self.labels[0], self)
        self.label_2 = QLabel(self.labels[1], self)
        self.label_3 = QLabel(self.labels[2], self)
        self.label_4 = QLabel(self.labels[3], self)
        self.label_5 = QLabel(self.labels[4], self)
        self.label_6 = QLabel(self.labels[5], self)
        self.label_7 = QLabel(self.labels[6], self)
        self.label_8 = QLabel(self.labels[7], self)
        self.label_9 = QLabel(self.labels[8], self)
        self.label_10 = QLabel(self.labels[9], self)
    
    
    def create_LCD_numbers(self):
        self.num_1 = QLCDNumber(self)
        self.num_2 = QLCDNumber(self)
        self.num_3 = QLCDNumber(self)
        self.num_4 = QLCDNumber(self)
        self.num_5 = QLCDNumber(self)
        self.num_6 = QLCDNumber(self)
        self.num_7 = QLCDNumber(self)
        self.num_8 = QLCDNumber(self)
        self.num_9 = QLCDNumber(self)
        self.num_10 = QLCDNumber(self)
    
    
    def create_sliders(self):
        self.sld_1 = QSlider(Qt.Vertical, self)
        self.sld_2 = QSlider(Qt.Vertical, self)
        self.sld_3 = QSlider(Qt.Vertical, self)
        self.sld_4 = QSlider(Qt.Vertical, self)
        self.sld_5 = QSlider(Qt.Vertical, self)
        self.sld_6 = QSlider(Qt.Vertical, self)
        self.sld_7 = QSlider(Qt.Vertical, self)
        self.sld_8 = QSlider(Qt.Vertical, self)
        self.sld_9 = QSlider(Qt.Vertical, self)
        self.sld_10 = QSlider(Qt.Vertical, self)
        
        self.sld_1.setMinimum(self.sld_min)
        self.sld_2.setMinimum(self.sld_min)
        self.sld_3.setMinimum(self.sld_min)
        self.sld_4.setMinimum(self.sld_min)
        self.sld_5.setMinimum(self.sld_min)
        self.sld_6.setMinimum(self.sld_min)
        self.sld_7.setMinimum(self.sld_min)
        self.sld_8.setMinimum(self.sld_min)
        self.sld_9.setMinimum(self.sld_min)
        self.sld_10.setMinimum(self.sld_min)
        
        self.sld_1.setMaximum(self.sld_max)
        self.sld_2.setMaximum(self.sld_max)
        self.sld_3.setMaximum(self.sld_max)
        self.sld_4.setMaximum(self.sld_max)
        self.sld_5.setMaximum(self.sld_max)
        self.sld_6.setMaximum(self.sld_max)
        self.sld_7.setMaximum(self.sld_max)
        self.sld_8.setMaximum(self.sld_max)
        self.sld_9.setMaximum(self.sld_max)
        self.sld_10.setMaximum(self.sld_max)
        
        self.sld_1.setValue(self.sld_def)
        self.sld_2.setValue(self.sld_def)
        self.sld_3.setValue(self.sld_def)
        self.sld_4.setValue(self.sld_def)
        self.sld_5.setValue(self.sld_def)
        self.sld_6.setValue(self.sld_def)
        self.sld_7.setValue(self.sld_def)
        self.sld_8.setValue(self.sld_def)
        self.sld_9.setValue(self.sld_def)
        self.sld_10.setValue(self.sld_def)
        
        self.sld_1.setFocusPolicy(Qt.StrongFocus)
        self.sld_2.setFocusPolicy(Qt.StrongFocus)
        self.sld_3.setFocusPolicy(Qt.StrongFocus)
        self.sld_4.setFocusPolicy(Qt.StrongFocus)
        self.sld_5.setFocusPolicy(Qt.StrongFocus)
        self.sld_6.setFocusPolicy(Qt.StrongFocus)
        self.sld_7.setFocusPolicy(Qt.StrongFocus)
        self.sld_8.setFocusPolicy(Qt.StrongFocus)
        self.sld_9.setFocusPolicy(Qt.StrongFocus)
        self.sld_10.setFocusPolicy(Qt.StrongFocus)
        
        self.sld_1.setTickPosition(QSlider.TicksBothSides)
        self.sld_2.setTickPosition(QSlider.TicksBothSides)
        self.sld_3.setTickPosition(QSlider.TicksBothSides)
        self.sld_4.setTickPosition(QSlider.TicksBothSides)
        self.sld_5.setTickPosition(QSlider.TicksBothSides)
        self.sld_6.setTickPosition(QSlider.TicksBothSides)
        self.sld_7.setTickPosition(QSlider.TicksBothSides)
        self.sld_8.setTickPosition(QSlider.TicksBothSides)
        self.sld_9.setTickPosition(QSlider.TicksBothSides)
        self.sld_10.setTickPosition(QSlider.TicksBothSides)
        
        self.sld_1.setTickInterval(self.sld_interval)
        self.sld_2.setTickInterval(self.sld_interval)
        self.sld_3.setTickInterval(self.sld_interval)
        self.sld_4.setTickInterval(self.sld_interval)
        self.sld_5.setTickInterval(self.sld_interval)
        self.sld_6.setTickInterval(self.sld_interval)
        self.sld_7.setTickInterval(self.sld_interval)
        self.sld_8.setTickInterval(self.sld_interval)
        self.sld_9.setTickInterval(self.sld_interval)
        self.sld_10.setTickInterval(self.sld_interval)
        
        self.sld_1.setSingleStep(self.sld_step)
        self.sld_2.setSingleStep(self.sld_step)
        self.sld_3.setSingleStep(self.sld_step)
        self.sld_4.setSingleStep(self.sld_step)
        self.sld_5.setSingleStep(self.sld_step)
        self.sld_6.setSingleStep(self.sld_step)
        self.sld_7.setSingleStep(self.sld_step)
        self.sld_8.setSingleStep(self.sld_step)
        self.sld_9.setSingleStep(self.sld_step)
        self.sld_10.setSingleStep(self.sld_step)
        
        self.sld_1.valueChanged[int].connect(self.sliderChangeValue)
        self.sld_2.valueChanged[int].connect(self.sliderChangeValue)
        self.sld_3.valueChanged[int].connect(self.sliderChangeValue)
        self.sld_4.valueChanged[int].connect(self.sliderChangeValue)
        self.sld_5.valueChanged[int].connect(self.sliderChangeValue)
        self.sld_6.valueChanged[int].connect(self.sliderChangeValue)
        self.sld_7.valueChanged[int].connect(self.sliderChangeValue)
        self.sld_8.valueChanged[int].connect(self.sliderChangeValue)
        self.sld_9.valueChanged[int].connect(self.sliderChangeValue)
        self.sld_10.valueChanged[int].connect(self.sliderChangeValue)
        
        self.sld_1.valueChanged[int].connect(self.num_1.display)
        self.sld_2.valueChanged[int].connect(self.num_2.display)
        self.sld_3.valueChanged[int].connect(self.num_3.display)
        self.sld_4.valueChanged[int].connect(self.num_4.display)
        self.sld_5.valueChanged[int].connect(self.num_5.display)
        self.sld_6.valueChanged[int].connect(self.num_6.display)
        self.sld_7.valueChanged[int].connect(self.num_7.display)
        self.sld_8.valueChanged[int].connect(self.num_8.display)
        self.sld_9.valueChanged[int].connect(self.num_9.display)
        self.sld_10.valueChanged[int].connect(self.num_10.display)
        
        self.old_value_sld1 = self.sld_def
        self.old_value_sld2 = self.sld_def
        self.old_value_sld3 = self.sld_def
        self.old_value_sld4 = self.sld_def
        self.old_value_sld5 = self.sld_def
        self.old_value_sld6 = self.sld_def
        self.old_value_sld7 = self.sld_def
        self.old_value_sld8 = self.sld_def
        self.old_value_sld9 = self.sld_def
        self.old_value_sld10 = self.sld_def
        
    
    def create_checkboxes(self):
        self.checkbox_1 = QCheckBox(self.checkboxes_lables[0], self)
        self.checkbox_2 = QCheckBox(self.checkboxes_lables[1], self)
        
        self.checkbox_1.setChecked(False)
        self.checkbox_2.setChecked(False)
        
        self.checkbox_1.stateChanged.connect(self.checkboxClicked)
        self.checkbox_2.stateChanged.connect(self.checkboxClicked)
        
        
    def create_buttons(self):
        self.btn_1 = QPushButton(self.btns_lables[0], self)
        self.btn_2 = QPushButton(self.btns_lables[1], self)
        self.btn_3 = QPushButton(self.btns_lables[2], self)
        
        self.btn_1.clicked.connect(self.buttonClicked)
        self.btn_2.clicked.connect(self.buttonClicked)
        self.btn_3.clicked.connect(self.buttonClicked)
        
        
    def create_graphics(self):
        figure_1 = plt.figure()
        self.figure_2 = plt.figure()
        self.figure_4 = plt.figure()
        
        self.canvas_1 = FigureCanvas(figure_1)
        self.canvas_2 = FigureCanvas(self.figure_2)
        self.canvas_4 = FigureCanvas(self.figure_4)
            
        self.toolbar_1 = NavigationToolbar(self.canvas_1, self)
        self.toolbar_2 = NavigationToolbar(self.canvas_2, self)
        self.toolbar_4 = NavigationToolbar(self.canvas_4, self)

        figure_1.clear()
        self.figure_2.clear()
        self.figure_4.clear()
        
        ax_1 = figure_1.add_subplot(1, 1, 1)
        self.ax_2 = self.figure_2.add_subplot(1, 1, 1)
        self.ax_4 = self.figure_4.add_subplot(1, 1, 1)
        
        ax_1.set_xlabel('Частота, Гц')
        self.ax_2.set_xlabel('Частота, Гц')
        self.ax_4.set_xlabel('Время, с')
        
        figure_1.align_xlabels()
        self.figure_2.align_xlabels()
        self.figure_4.align_xlabels()
        
        ax_1.set_ylabel('Амплитуда')
        self.ax_2.set_ylabel('Амплитуда')
        self.ax_4.set_ylabel('Амплитуда')
        
        figure_1.align_ylabels()
        self.figure_2.align_ylabels()
        self.figure_4.align_ylabels()
    
        self.elem_per_herz = self.spectrum.shape[1] // (self.max_freq - self.min_freq)
        
        ax_1.plot(np.fft.rfftfreq(self.nframes, 1./ self.framerate)[::self.koeff], 
                  np.abs(self.spectrum[0][::self.koeff]) / self.nframes)
        self.ax_2.plot(np.fft.rfftfreq(self.nframes, 1./ self.framerate)[::self.koeff],
                       np.abs(self.spectrum[0][::self.koeff]) / self.nframes)
        self.ax_4.plot(self.channels[0][::self.koeff])
        
        self.canvas_1.draw()
        self.canvas_2.draw()
        self.canvas_4.draw()

        
    def create_interface(self):
        self.labels_box = QHBoxLayout()
        self.labels_box.addWidget(self.label_1)
        self.labels_box.addWidget(self.label_2)
        self.labels_box.addWidget(self.label_3)
        self.labels_box.addWidget(self.label_4)
        self.labels_box.addWidget(self.label_5)
        self.labels_box.addWidget(self.label_6)
        self.labels_box.addWidget(self.label_7)
        self.labels_box.addWidget(self.label_8)
        self.labels_box.addWidget(self.label_9)
        self.labels_box.addWidget(self.label_10)
        
        self.nums_box = QHBoxLayout()
        self.nums_box.addWidget(self.num_1)
        self.nums_box.addWidget(self.num_2)
        self.nums_box.addWidget(self.num_3)
        self.nums_box.addWidget(self.num_4)
        self.nums_box.addWidget(self.num_5)
        self.nums_box.addWidget(self.num_6)
        self.nums_box.addWidget(self.num_7)
        self.nums_box.addWidget(self.num_8)
        self.nums_box.addWidget(self.num_9)
        self.nums_box.addWidget(self.num_10)
        
        self.slds_box = QHBoxLayout()
        self.slds_box.addWidget(self.sld_1)
        self.slds_box.addWidget(self.sld_2)
        self.slds_box.addWidget(self.sld_3)
        self.slds_box.addWidget(self.sld_4)
        self.slds_box.addWidget(self.sld_5)
        self.slds_box.addWidget(self.sld_6)
        self.slds_box.addWidget(self.sld_7)
        self.slds_box.addWidget(self.sld_8)
        self.slds_box.addWidget(self.sld_9)
        self.slds_box.addWidget(self.sld_10)

        self.graphs_box_1 = QVBoxLayout()
        self.graphs_box_1.addWidget(self.toolbar_4)
        self.graphs_box_1.addWidget(self.canvas_4)
        
        self.checks_and_btns_box = QHBoxLayout()
        self.checks_and_btns_box.addWidget(self.checkbox_1)
        self.checks_and_btns_box.addWidget(self.checkbox_2)
        self.checks_and_btns_box.addWidget(self.btn_1)
        self.checks_and_btns_box.addWidget(self.btn_2)
        self.checks_and_btns_box.addWidget(self.btn_3)
        
        self.graphs_box_2 = QVBoxLayout()
        self.graphs_box_2.addWidget(self.toolbar_1)
        self.graphs_box_2.addWidget(self.canvas_1)
        self.graphs_box_2.addWidget(self.toolbar_2)
        self.graphs_box_2.addWidget(self.canvas_2)
        
        self.left_box = QVBoxLayout()
        self.left_box.addLayout(self.labels_box)
        self.left_box.addLayout(self.slds_box)
        self.left_box.addLayout(self.nums_box)
        self.left_box.addLayout(self.graphs_box_1)
        
        self.right_box = QVBoxLayout()
        self.right_box.addLayout(self.checks_and_btns_box)
        self.right_box.addLayout(self.graphs_box_2)
        
        self.all_box = QHBoxLayout()
        self.all_box.addLayout(self.left_box)
        self.all_box.addLayout(self.right_box)
        
        self.setLayout(self.all_box)
        
        self.setWindowTitle(self.app_name)
        self.showMaximized()
        
        
    def sliderChangeValue(self, value):
        if (self.sender() == self.sld_1):
            self.sld1_worker = Worker(self.music_edit, 0, value)
            self.threadpool.start(self.sld1_worker)
            
        elif (self.sender() == self.sld_2):
            self.sld2_worker = Worker(self.music_edit, 1, value)
            self.threadpool.start(self.sld2_worker)
            
        elif (self.sender() == self.sld_3):
            self.sld3_worker = Worker(self.music_edit, 2, value)
            self.threadpool.start(self.sld3_worker)
            
        elif (self.sender() == self.sld_4):
            self.sld4_worker = Worker(self.music_edit, 3, value)
            self.threadpool.start(self.sld4_worker)
            
        elif (self.sender() == self.sld_5):
            self.sld5_worker = Worker(self.music_edit, 4, value)
            self.threadpool.start(self.sld5_worker)
            
        elif (self.sender() == self.sld_6):
            self.sld6_worker = Worker(self.music_edit, 5, value)
            self.threadpool.start(self.sld6_worker)
            
        elif (self.sender() == self.sld_7):
            self.sld7_worker = Worker(self.music_edit, 6, value)
            self.threadpool.start(self.sld7_worker)
            
        elif (self.sender() == self.sld_8):
            self.sld8_worker = Worker(self.music_edit, 7, value)
            self.threadpool.start(self.sld8_worker)
            
        elif (self.sender() == self.sld_9):
            self.sld9_worker = Worker(self.music_edit, 8, value)
            self.threadpool.start(self.sld9_worker)
            
        else:
            self.sld10_worker = Worker(self.music_edit, 9, value)
            self.threadpool.start(self.sld10_worker)
        
            
    def checkboxClicked(self, state):
        if (self.sender() == self.checkbox_1):
            if (state == Qt.Checked):
                self.checkbox_2.setChecked(False)
                self.channels = self.channels_kliping.copy()
                self.spectrum = self.spectrum_kliping.copy()
            else:
                self.channels = self.channels_original.copy()
                self.spectrum = self.spectrum_original.copy()
                
        else:
            if (state == Qt.Checked):
                self.checkbox_1.setChecked(False)
                self.channels = self.channels_envelop.copy()
                self.spectrum = self.spectrum_envelop.copy()
            else:
                self.channels = self.channels_original.copy()
                self.spectrum = self.spectrum_original.copy()
                
        self.sld_1.setValue(self.sld_def)
        self.sld_2.setValue(self.sld_def)
        self.sld_3.setValue(self.sld_def)
        self.sld_4.setValue(self.sld_def)
        self.sld_5.setValue(self.sld_def)
        self.sld_6.setValue(self.sld_def)
        self.sld_7.setValue(self.sld_def)
        self.sld_8.setValue(self.sld_def)
        self.sld_9.setValue(self.sld_def)
        self.sld_10.setValue(self.sld_def)
        
        draw_1 = Worker(self.draw_array, self.spectrum, 0)
        self.threadpool.start(draw_1)

        draw_2 = Worker(self.draw_array, self.channels, 1)
        self.threadpool.start(draw_2)
            
        
    def buttonClicked(self):
        if (self.sender() == self.btn_1):
            if (self.music_is_playing == False):
                self.music_is_playing = True
                self.music_worker = Worker(self.start_music)
                self.threadpool.start(self.music_worker)
            
        elif (self.sender() == self.btn_2):
            if (self.music_is_playing == True):
                self.music_is_playing = False
            
        else:
            if (self.music_is_playing == True):
                self.music_is_playing = False
                self.threadpool.clear()
                
            sliders = [self.sld1_worker, self.sld2_worker,
                       self.sld3_worker, self.sld4_worker,
                       self.sld5_worker, self.sld6_worker,
                       self.sld7_worker, self.sld8_worker,
                       self.sld9_worker, self.sld10_worker]
            for slider in sliders:
                self.sld_stop(slider)
            
            self.buffer_cnt = 0
            
            self.sld_1.setValue(self.sld_def)
            self.sld_2.setValue(self.sld_def)
            self.sld_3.setValue(self.sld_def)
            self.sld_4.setValue(self.sld_def)
            self.sld_5.setValue(self.sld_def)
            self.sld_6.setValue(self.sld_def)
            self.sld_7.setValue(self.sld_def)
            self.sld_8.setValue(self.sld_def)
            self.sld_9.setValue(self.sld_def)
            self.sld_10.setValue(self.sld_def)
            
            self.checkbox_1.setChecked(False)
            self.checkbox_2.setChecked(False)
            
            tmp_worker = Worker(self.tmp_func)
            self.threadpool.start(tmp_worker)
            
    
    def sld_stop(self, slider):
        ids = { self.sld1_worker: 0, self.sld2_worker: 1,
                self.sld3_worker: 2, self.sld4_worker: 3,
                self.sld5_worker: 4, self.sld6_worker: 5,
                self.sld7_worker: 6, self.sld8_worker: 7,
                self.sld9_worker: 8, self.sld10_worker: 9}
        
        slider = Worker(self.music_edit, ids[slider], self.sld_def)
        self.threadpool.start(slider)
        
        
    def tmp_func(self):
        while (self.threadpool.activeThreadCount() != 1):
            sleep(0.1)
        self.channels = self.channels_original.copy()
        self.spectrum = self.spectrum_original.copy()
        print('music stopped')


    def start_music(self):
            tmp_channels = []
            tmp_channels.append(self.channels[0][self.buffer_cnt * self.buffer_size:
                                             (self.buffer_cnt + 1) * self.buffer_size + 1:])
            tmp_channels.append(self.channels[1][self.buffer_cnt * self.buffer_size:
                                             (self.buffer_cnt + 1) * self.buffer_size + 1:])
            tmp_channels = np.array(tmp_channels)
            tmp_channels = np.ascontiguousarray(tmp_channels.T)
            tmp_sound = pygame.sndarray.make_sound(tmp_channels)
            
            sound = tmp_sound
            if (self.music_is_playing == False):
                return
            pygame.mixer.Sound.play(sound)
        
            start_pos = self.buffer_cnt
            for self.buffer_cnt in range(start_pos + 1, self.nframes // self.buffer_size):
                tmp_channels = []
                tmp_channels.append(self.channels[0][self.buffer_cnt * self.buffer_size:
                                                     (self.buffer_cnt + 1) * self.buffer_size + 1:])
                tmp_channels.append(self.channels[1][self.buffer_cnt * self.buffer_size:
                                                     (self.buffer_cnt + 1) * self.buffer_size + 1:])
                tmp_channels = np.array(tmp_channels)
                tmp_channels = np.ascontiguousarray(tmp_channels.T)
                tmp_sound = pygame.sndarray.make_sound(tmp_channels)
            
                while (pygame.mixer.get_busy()):
                    sleep(0.01)
                    
                sound = tmp_sound
                if (self.music_is_playing == False):
                    return
                pygame.mixer.Sound.play(sound)
        
            tmp_channels = []
            tmp_channels.append(self.channels[0][self.buffer_cnt * self.buffer_size::])
            tmp_channels.append(self.channels[1][self.buffer_cnt * self.buffer_size::])
            tmp_channels = np.array(tmp_channels)
            tmp_channels = np.ascontiguousarray(tmp_channels.T)
            tmp_sound = pygame.sndarray.make_sound(tmp_channels)
        
            while (pygame.mixer.get_busy()):
                sleep(0.01)
             
            sound = tmp_sound
            if (self.music_is_playing == False):
                return
            pygame.mixer.Sound.play(sound)
            
            self.buffer_cnt = 0
            self.music_is_playing = False
            
            
    def music_edit(self, pos, value):
        old_values = {
            0: self.old_value_sld1,
            1: self.old_value_sld2,
            2: self.old_value_sld3,
            3: self.old_value_sld4,
            4: self.old_value_sld5,
            5: self.old_value_sld6,
            6: self.old_value_sld7,
            7: self.old_value_sld8,
            8: self.old_value_sld9,
            9: self.old_value_sld10
        }
        old_value = old_values[pos]
        
        if pos == 0:
            self.old_value_sld1 = value
        elif pos == 1:
            self.old_value_sld2 = value
        elif pos == 2:
            self.old_value_sld3 = value
        elif pos == 3:
            self.old_value_sld4 = value
        elif pos == 4:
            self.old_value_sld5 = value
        elif pos == 5:
            self.old_value_sld6 = value
        elif pos == 6:
            self.old_value_sld7 = value
        elif pos == 7:
            self.old_value_sld8 = value
        elif pos == 8:
            self.old_value_sld9 = value
        else:
            self.old_value_sld10 = value
        
        if (old_value == value):
            return
        
        if (pos == 0):
            for i in range(self.nchannels):
                self.spectrum[i][:self.elem_per_herz * self.bands[1][pos] + 1] *= 10**((value - old_value) / 20)
                
        elif (pos == 9):
            for i in range(self.nchannels):
                self.spectrum[i][self.elem_per_herz * self.bands[0][pos]:] *= 10**((value - old_value) / 20)
                
        else:
            for i in range(self.nchannels):
                self.spectrum[i][self.elem_per_herz * self.bands[0][pos]:self.elem_per_herz * self.bands[1][pos] +
                                 1] *= 10**((value - old_value) / 20)
        
        self.channels = (np.fft.irfft(self.spectrum)).astype(self.types[self.sampwidth])

        draw_1 = Worker(self.draw_array, self.spectrum, 0)
        self.threadpool.start(draw_1)

        draw_2 = Worker(self.draw_array, self.channels, 1)
        self.threadpool.start(draw_2)
        
        
    def draw_array(self, arr, spectrum_or_channel):
        if (spectrum_or_channel == 0):
            self.figure_2.clear()
            self.ax_2 = self.figure_2.add_subplot(1, 1, 1)
            self.ax_2.set_xlabel('Частота, Гц')
            self.figure_2.align_xlabels()
            self.ax_2.set_ylabel('Амплитуда')
            self.figure_2.align_ylabels()
            self.ax_2.plot(np.fft.rfftfreq(self.nframes, 1./ self.framerate)[::self.koeff],
                       np.abs(arr[0][::self.koeff]) / self.nframes)
            self.canvas_2.draw()
            
        else:
            self.figure_4.clear()
            self.ax_4 = self.figure_4.add_subplot(1, 1, 1)
            self.ax_4.set_xlabel('Время, с')
            self.figure_4.align_xlabels()
            self.ax_4.set_ylabel('Амплитуда')
            self.figure_4.align_ylabels()
            self.ax_4.plot(arr[0][::self.koeff])
            self.canvas_4.draw()
    
    
    def doing_kliping(self, channels):
        print('kliping start')
        start_time = time()
        threshold_max = int(0.6 * np.max(channels[0]))
        threshold_min = int(0.6 * np.min(channels[0]))
        
        self.channels_kliping = np.maximum(np.minimum(channels, threshold_max),
                                           threshold_min).astype(self.types[self.sampwidth])
        self.spectrum_kliping = np.fft.rfft(self.channels_kliping)
        print('kliping end: ' + str(time() - start_time))
    
    
    def doing_envelop(self, channels):
        print('envelop start')
        start_time = time()
        frequency = 1 / 15
        envelope_sig = np.array([abs(sin(2 * pi * frequency * t / self.framerate)) 
                                 for t in range(self.nframes)])
        tmp_channels = channels.copy()
        
        for i in range(self.nchannels):
                tmp_channels[i] = (tmp_channels[i] * envelope_sig).astype(self.types[self.sampwidth])
        
        self.channels_envelop = tmp_channels.copy()
        self.spectrum_envelop = np.fft.rfft(self.channels_envelop)
        print('envelop end: ' + str(time() - start_time))


    
if __name__ == '__main__':
    Equalizer = QApplication(sys.argv)
    Main_Window = Main_Window_class()
    sys.exit(Equalizer.exec_())

