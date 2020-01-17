#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 10:34:49 2020

@author: jrkjithin
"""
from math import pi
import librosa as lr
import librosa.display as ld
from scipy import fft
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import IPython.display as ipd

audio_files = glob("./gunshot/*.wav")
len(audio_files)
for i in range(len(audio_files)):
  audio,sr = lr.load(audio_files[i])
  time = np.arange(0,len(audio))/sr
  n_fft = 1024
  f=20
  x=np.sin(2*pi*f*time)
#  plt.subplot(2,1,1)
#  plt.plot(time,audio)
#  plt.xlabel("Time(s)")
 # plt.ylabel("Amplitude")
  n=np.size(time)
  fr = (sr/2)*np.linspace(0,1,n/2)
  X = fft(audio)
  X_m = (2/n)*abs(X[0:np.size(fr)])
  plt.subplot(2,1,1)
  plt.plot(fr,X_m)
  plt.xlabel("Frequency")
  plt.ylabel("Amplitude")
  plt.tight_layout()
  plt.axis('off')
  plt.savefig("./gunshotimg/fft"+str(i)+".png",bbox_inches='tight')
  plt.close()

