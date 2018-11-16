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
from scipy.io import loadmat
from numpy.matlib import repmat
import argparse

class Gammatone :
      
      def __init__(self,samplerate, nFilterBanks):
            self.filterOrder  = 4;
            self.filterLength = 512;
            self.nFilterBanks = nFilterBanks;
            self.samplerate   = samplerate;
            self.fRange       = [30, samplerate]
            self.gFilters     = self.gammtone_filters()
      
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
            b = 1.019*24.7*(4.37*center_freq/1000+1);     
               
            gFilters = np.zeros((self.filterLength, self.nFilterBanks))
            tmp_t = np.array(range(0,self.filterLength))/self.samplerate;
            gain = (10**((self.loudness(center_freq)-60)/20)/3)*((2*np.pi*b/self.samplerate)**4); 
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", required=True, action='store_true', help="Degraded audio file (absolute path)")
    parser.add_argument("-o", required=True, action='store_true', help="Enhanced audio file (absolute path)")
    parser.add_argument("-n", required=False, action='store_true', help="Number of Gammtone FilterBanks")
    parser.add_argument("-w", required=False, action='store_true', help="Window Length (ms)")
    parser.add_argument("-s", required=False, action='store_true', help="Overlap (ms)")

    args = parser.parse_args()
    gt = Gammatone(64, 8000)
