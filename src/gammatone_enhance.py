#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 15:33:35 2019

@author: VinayKothapally

"""
import os
import numpy as np
import wpe as wpe
import logmmse as mmse
import gammatone as gt
import soundfile as sf
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt



class Audio :
    def __init__(self,speechFile):
        self.speechPath = speechFile
        self.samplerate = sf.info(speechFile).samplerate
        self.blocksize  = 5*self.samplerate
        self.framesize  = 25e-3
        self.framehop   = 10e-3
        self.gamma_param = gt.get_GTfilters(64, self.samplerate, [30,int(0.5*self.samplerate)], self.framesize , self.framehop)
            
    def blocks(self):
        frames = [frame for frame in sf.blocks(self.speechPath, self.blocksize)]
        if np.shape(frames[-1:])[1] < 4*self.samplerate:
            temp = np.concatenate(frames[-2:])
            frames = frames[:len(frames)-2]
            frames.append(temp)
        return frames
    










def signal_enhance(speechFile, destFile, vadFile):
    dest_dir = '/'.join(destFile.split('/')[:-1])
    os.makedirs(dest_dir, exist_ok=True)
    vad_dir = '/'.join(vadFile.split('/')[:-1])
    os.makedirs(vad_dir, exist_ok=True)
      
      
    samplerate = sf.info(speechFile).samplerate
    
    x_wpe = []
    for block in sf.blocks(speechFile, 5*samplerate):
        x_wpe.append(wpe.wpe_dereverb(block, samplerate))
    return x_wpe
'''        
    # GammaTone Time-Frequnecy Mask Estimation
#    gamma_param = gt.get_GTfilters(64, samplerate, [30,int(0.5*samplerate)], 20e-3, 10e-3)
      
#    vad = np.zeros((nSamples,));
#    enhanced = np.zeros((nSamples,));
      
    rms = [np.sqrt(np.mean(block**2)) for block in sf.blocks(speechFile, blocksize=512, overlap=512)]

    return rms

#        enhanced[start:stop], vad[start:stop] = gt.GT_SPP_Enhance(block,gamma_param)

          

      
      if start<nSamples and stop > nSamples:
            signal = x_wpe[start:nSamples]
            enhanced[start:], vad[start:] = gt.GT_SPP_Enhance(signal,gamma_param)
      vad = savgol_filter(1.0*vad, int(0.5*samplerate)-1, 3)>0.5;
      enhanced = np.max(abs(x))*enhanced/np.max(abs(enhanced))
      
      
      # LOG-MMSE Enhancement
      enhanced_mmse = mmse.logmmse(enhanced, gamma_param['fs'])
      
      
      # write Audio to Disk
      sf.write(destFile, enhanced_mmse, samplerate)
      
      
      
      # Write VAD to Disk
      timings = np.argwhere(np.diff(vad)).squeeze()
      start_times = np.insert(timings+1, 0, 0)
      end_times = np.insert(timings,len(timings), len(vad))
      with open(vadFile,'w') as f:
            for k in range(len(start_times)):
                  if all(vad[start_times[k]:end_times[k]]):
                        word = 'speech'
                  else:
                        word = 'non-speech'
                  f.write(str(start_times[k]/samplerate) + ' ' + str(end_times[k]/samplerate) + ' ' + word+'\n')
      f.close()
'''


if __name__ == "__main__":
    speechFile = '/Users/vinaykothapally/GitHub/Speech-Dereverberation/audiofiles/Clean_long.wav'
    destFile = '/Users/vinaykothapally/GitHub/Speech-Dereverberation/src/Enhanced.wav'
    vadFile = '/Users/vinaykothapally/GitHub/Speech-Dereverberation/src/Enhanced_VAD.txt'
    
    audio = Audio(speechFile)
    x_wpe = []
    for block in audio.blocks():     #signal_enhance(speechFile, destFile, vadFile)
        x_wpe.append(wpe.wpe_dereverb(block, audio.samplerate))
