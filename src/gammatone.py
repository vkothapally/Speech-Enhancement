#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 16:43:56 2018

@author: Vinay Kothapally

Single-Channel Speech Deverberation using Gammtone FilterBank
This method Uses:    
    * Gammtone FilterBanks
    * Time-Frequency Masking 
    * Spectral Subtraction
    * MMSE/log-MMSE enahncer
    * Post-Filtering

"""

import numpy as np
import soundfile as sf
from scipy.io import loadmat
from scipy.signal import lfilter, spectrogram
from numpy.matlib import repmat
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class Gammatone :
      
      def __init__(self,samplerate, nFilterBanks):
            self.filterOrder  = 4;
            self.filterLength = 512;
            self.nFilterBanks = nFilterBanks;
            self.samplerate   = samplerate;
            self.fRange       = [30, samplerate]
            self.gFilters     = self.gammtone_filters()
            self.pca          = PCA(n_components=1)
            self.scaler       = StandardScaler()
            
      def hz2erb(self, frequecny):
            hz = np.asarray(frequecny)
            return 21.4*np.log10(4.37e-3*hz+1)
      
      def erb2hz(self, frequecny):
            erb = np.asarray(frequecny)
            return (10**(erb/21.4)-1)/4.37e-3
            
      def load_coeff(self, filename):
            coeff = loadmat('f_af_bf_cf.mat')
            af = coeff['af'].flatten()
            bf = coeff['bf'].flatten()
            cf = coeff['cf'].flatten()
            ff = coeff['ff'].flatten()
            return af, bf, cf, ff
      
      def loudness(self,frequecny):  
            freq = np.asarray(frequecny)
            dB   = 60
            af, bf, cf, ff = self.load_coeff('f_af_bf_cf.mat')
            if any(freq<20) or any(freq>12500):
                  print('Accepted frequency range: [20,12500]')
                  return
            idx = np.array([np.max(np.where(ff<freq[k])) for k in range(len(freq))]).flatten()
            afy=af[idx]+(freq-ff[idx])*(af[idx+1]-af[idx])/(ff[idx+1]-ff[idx]);
            bfy=bf[idx]+(freq-ff[idx])*(bf[idx+1]-bf[idx])/(ff[idx+1]-ff[idx]);
            cfy=cf[idx]+(freq-ff[idx])*(cf[idx+1]-cf[idx])/(ff[idx+1]-ff[idx]);
            loud =4.2+afy*(dB-cfy)/(1+bfy*(dB-cfy));      
            return loud 
      
      def gammtone_filters(self):
            erb_b = self.hz2erb(self.fRange)
            erb = np.linspace(erb_b[0], erb_b[1], self.nFilterBanks, endpoint=True)
            center_freq = self.erb2hz(erb);
            b = 1.019*24.7*(4.37*center_freq/1000+1)     
            self.midearcoeff = 10**((self.loudness(center_freq)-60)/20)    
            gFilters = np.zeros((self.filterLength, self.nFilterBanks))
            tmp_t = np.array(range(0,self.filterLength))/self.samplerate;
            gain = (self.midearcoeff/3)*((2*np.pi*b/self.samplerate)**4); 
            for k in range(self.nFilterBanks):
                  gFilters[:,k] = gain[k]*(self.samplerate**3)*(tmp_t**(self.filterOrder-1))*\
                                    np.exp(-2*np.pi*b[k]*tmp_t)*np.cos(2*np.pi*center_freq[k]*tmp_t)
            return gFilters
      
      def get_frames(self, signal, framelen, overlap):
            numframes = int(np.floor(len(signal)/overlap)-1)
            frame_idx = repmat(np.array(range(framelen))[:, np.newaxis],1,numframes) + \
                        repmat(np.array(range(numframes))[np.newaxis,:]*overlap,framelen,1)
            if frame_idx[-1,-1]<len(signal)-1:
                  samples = len(signal) - frame_idx[-1,-1]
                  last_frame = np.zeros((framelen,1))
                  last_frame[0:samples,0] = numframes*overlap+np.array(range(samples))
                  frame_idx = np.hstack((frame_idx, last_frame))
                  numframes = numframes + 1
            return frame_idx.astype(int), numframes
      
      def compute_feats(self, signal, frame_idx):
            subband_frames = signal[frame_idx]
            energy = np.array(np.abs(np.sum(subband_frames**2, axis=0))) 
            pscore = np.array(np.var(subband_frames, axis=0)**2.1/np.var(np.abs(subband_frames),axis=0))
            diff_n = np.array(np.var([(np.mean(np.diff(subband_frames,n, axis=0),axis=0)) for n in range(5)],axis=0))
            corr_n = np.mean(np.vstack([np.append(list(np.zeros((n))), np.mean(subband_frames[:,:-n]*subband_frames[:,n:],axis=0)) for n in range(1,6)]), axis=0)
            features = np.vstack([energy, pscore, diff_n, corr_n]).transpose()
            features = self.pca.fit_transform(self.scaler.fit_transform(features)).flatten()
            features = self.softmax(1/(1 + np.exp(-1.2*features)))
            return features
            
      def softmax(self, input):
            output = np.exp(input)/np.sum(np.exp(input)+1e-3)
            output = (output - np.min(output))/np.max(output)
            return output

      def gammatoneMaskEstimate(self, block, framelen, overlap):
            subband_signals = np.vstack([lfilter(self.gFilters[:,k],1,block) for k in range(self.nFilterBanks)]).transpose()
            frame_idx, numframes = self.get_frames(block, framelen, overlap)
            Mask = np.zeros((self.nFilterBanks,numframes))
            for i in range(self.nFilterBanks):
                  Mask[i,:] = self.compute_feats(subband_signals[:,i], frame_idx)
            Mask = np.flipud(Mask/repmat(np.max(Mask,axis=1)[:,np.newaxis],1,numframes))
            enhanced = self.gammatone_synthesis(subband_signals, Mask, block, framelen, overlap, frame_idx, numframes)
            enhanced = np.max(block)*enhanced/np.max(enhanced)
            return enhanced, Mask
      
      def gammatone_synthesis(self, subband_signals, Mask, block, framelen, overlap, frame_idx, numframes):
            coswin    = (1+np.cos(2*np.pi*np.array(range(framelen))/framelen- np.pi))/2
            subband_signals = np.flipud(subband_signals)/self.midearcoeff
            subband_signals = np.vstack([lfilter(self.gFilters[:,k],1,subband_signals[:,k]) for k in range(self.nFilterBanks)]).transpose()
            subband_signals = np.flipud(subband_signals)/self.midearcoeff
            weight = np.zeros((len(block), self.nFilterBanks))
            
            for m in range(numframes): 
                  weight[frame_idx[:,m],:] = weight[frame_idx[:,m],:] + np.transpose(np.matmul(Mask[:,m][:,np.newaxis],coswin[np.newaxis,:]))
            enhanced = np.mean(subband_signals*weight, axis=1)      
            return enhanced

audio, samplerate = sf.read('../audiofiles/Clean.wav')
gt = Gammatone(samplerate, 256)
framelen = int(20e-3*samplerate)
overlap = int(10e-3*samplerate)
output, Mask= gt.gammatoneMaskEstimate(audio,framelen, overlap)
sf.write('../audiofiles/Enhanced.wav', output, samplerate)

plt.figure(num=101)
f, t, Sxx = spectrogram(audio, samplerate, window=np.hamming(framelen), noverlap=overlap, scaling='spectrum',mode='magnitude')
plt.pcolormesh(t, f, 10*np.log10(Sxx))
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()



plt.figure(num=102)
f, t, Sxx = spectrogram(output, samplerate, window=np.hamming(framelen), noverlap=overlap, scaling='spectrum',mode='magnitude')
plt.pcolormesh(t, f, 10*np.log10(Sxx))
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

'''
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.figure()
ax = plt.gca()
im = ax.imshow(Mask, cmap='bone')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
'''



