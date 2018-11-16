#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 16:43:56 2018

@author: Vinay Kothapally
Gammtone FilterBank Analysis and Synthesis
"""

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
class Gammatone :
      
      def __init__(self):
            filterOrder  = 4;
            filterLength = 512;
            nFilterBanks = 64;
            samplerate   = 8000;
            fRange       = [30, samplerate]
            self.gFilters     = self.gammtone_filters(nFilterBanks, filterLength, filterOrder, samplerate, fRange)
      
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
      # Computes loudness level in Phons on the basis of equal-loudness functions.
            freq = np.asarray(frequecny)
            dB   = 60
            af, bf, cf, ff = self.load_coeff('f_af_bf_cf.mat')
            if any(freq<20) or any(freq>12500):
                  print('Accepted frequency range: [20,12500]')
                  return
            idx = np.array([find(ff<freq[k])[-1:] for k in range(len(freq))]).flatten()
            afy=af[idx]+(freq-ff[idx])*(af[idx+1]-af[idx])/(ff[idx+1]-ff[idx]);
            bfy=bf[idx]+(freq-ff[idx])*(bf[idx+1]-bf[idx])/(ff[idx+1]-ff[idx]);
            cfy=cf[idx]+(freq-ff[idx])*(cf[idx+1]-cf[idx])/(ff[idx+1]-ff[idx]);
            loud =4.2+afy*(dB-cfy)/(1+bfy*(dB-cfy));      
            return loud 
      
      def gammtone_filters(self,nFilterBanks, filterLength, filterOrder, samplerate, fRange):
            erb_b = self.hz2erb([30, 8000])
            erb = np.linspace(erb_b[0], erb_b[1], nFilterBanks, endpoint=True)
            center_freq = self.erb2hz(erb);
            b = 1.019*24.7*(4.37*center_freq/1000+1);     
               
            gFilters = np.zeros((filterLength, nFilterBanks))
            tmp_t = np.array(range(0,filterLength))/samplerate;
            gain = (10**((self.loudness(center_freq)-60)/20)/3)*((2*np.pi*b/samplerate)**4); 
            for k in range(nFilterBanks):
                  gFilters[:,k] = gain[k]*(samplerate**3)*(tmp_t**(filterOrder-1))*\
                                    np.exp(-2*np.pi*b[k]*tmp_t)*np.cos(2*np.pi*center_freq[k]*tmp_t)
            return gFilters


gt = Gammatone()
