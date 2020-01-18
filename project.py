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

audio_files = glob("../gunshot/*.wav")
len(audio_files)
for i in range(len(audio_files)):
  audio,sr = lr.load(audio_files[i], duration=1.0)
  #lr.output.write_wav("../gunshotimg/aud"+str(i)+".wav",audio,sr)
  time = np.arange(0,len(audio))/sr
  S = lr.feature.melspectrogram(y=audio, sr=sr, n_mels=128,
                                    fmax=(sr/2))

  plt.figure(figsize=(10, 4))
  S_dB = lr.power_to_db(S, ref=np.max)
  lr.display.specshow(S_dB, x_axis='time',y_axis='mel', sr=sr,fmax=(sr/2))
  plt.tight_layout()
  plt.axis('off')
  plt.savefig("../gunshotimg/mel"+str(i)+".png",dpi=1000,bbox_inches='tight')
  #plt.show()
  plt.close()

#ELEPHANT SOUND
audio_files = glob("../elephant/*.wav")
len(audio_files)
for i in range(len(audio_files)):
  audio,sr = lr.load(audio_files[i], duration=2.0)
  time = np.arange(0,len(audio))/sr
  S = lr.feature.melspectrogram(y=audio, sr=sr, n_mels=128,
                                    fmax=(sr/2))

  plt.figure(figsize=(10, 4))
  S_dB = lr.power_to_db(S, ref=np.max)
  lr.display.specshow(S_dB, x_axis='time',y_axis='mel', sr=sr,fmax=(sr/2))
  plt.tight_layout()
  plt.axis('off')
  plt.savefig("../elephantimg/mel"+str(i)+".png",dpi=1000,bbox_inches='tight')
  #plt.show()
  plt.close()
