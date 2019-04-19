#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 15:43:11 2019

@author: Vinay Kothapally


"""

import numpy as np
import scipy.io
import scipy.signal
from numpy import matlib as mb
from scipy.signal import medfilt2d, medfilt, resample
from sklearn.decomposition import PCA as sklearnPCA
np.seterr(under="ignore")

def hz2erb(hz):
      erb=21.4*np.log10(4.37e-3*hz+1)
      return np.array(erb)


def erb2hz(erb):
      hz=(10**(erb/21.4)-1)/4.37e-3
      return np.array(hz)



def loudness(freq):
      dB=60
      mat = scipy.io.loadmat('f_af_bf_cf.mat')
      af = mat['af'][0]
      bf = mat['bf'][0]
      cf = mat['cf'][0]
      ff = mat['ff'][0]
      
      if freq<20.0 or freq>12500.0:
            return 0
      k = 0
      while (ff[k] < freq):
            k=k+1
        
      afy=af[k-1]+(freq-ff[k-1])*(af[k]-af[k-1])/(ff[k]-ff[k-1])
      bfy=bf[k-1]+(freq-ff[k-1])*(bf[k]-bf[k-1])/(ff[k]-ff[k-1])
      cfy=cf[k-1]+(freq-ff[k-1])*(cf[k]-cf[k-1])/(ff[k]-ff[k-1])
      
      loud = (4.2+afy*(dB-cfy)/(1+bfy*(dB-cfy)))

      return loud
     
def nextpow2(x):
      """Return the first integer N such that 2**N >= abs(x)"""
      return np.ceil(np.log2(np.abs(x)))

def fftfilt_one(b, x, *n):
    """Filter the signal x with the FIR filter described by the
    coefficients in b using the overlap-add method. If the FFT
    length n is not specified, it and the overlap-add block length
    are selected so as to minimize the computational cost of
    the filtering operation."""
    
    N_x = len(x)
    N_b = len(b)

    # Determine the FFT length to use:
    if len(n):
        n = n[0]
        if n != np.int(n) or n <= 0:
            raise ValueError('n must be a nonnegative integer')
        if n < N_b:
            n = N_b
        N_fft = 2**nextpow2(n)
    else:

        if N_x > N_b:
            N = 2**np.arange(np.ceil(np.log2(N_b)),np.floor(np.log2(N_x)))
            cost = np.ceil(N_x/(N-N_b+1))*N*(np.log2(N)+1)
            N_fft = N[np.argmin(cost)]
        else:
            N_fft = 2**nextpow2(N_b+N_x-1)

    N_fft = int(N_fft)
    
    # Compute the block length:
    L = int(N_fft - N_b + 1) 
    # Compute the transform of the filter:
    H = np.fft.fft(b,N_fft)
    y = np.zeros(N_x,float)
    i = 0
    while i <= N_x:
        il = np.min([i+L,N_x])
        k = np.min([i+N_fft,N_x])
        yt = np.fft.ifft(np.fft.fft(x[i:il],N_fft)*H,N_fft) # Overlap..
        y[i:k] = y[i:k] + np.real(yt[:k-i])            # and add
        i += L
    return y

def fftfilt(b, x, *n):
    """Filter the signal x with the FIR filter described by the
    coefficients in b using the overlap-add method. If the FFT
    length n is not specified, it and the overlap-add block length
    are selected so as to minimize the computational cost of
    the filtering operation."""
    
    N_x = len(x)
    N_b = np.shape(b)[0]
    N_f = np.shape(b)[1]

    # Determine the FFT length to use:
    if len(n):
        n = n[0]
        if n != np.int(n) or n <= 0:
            raise ValueError('n must be a nonnegative integer')
        if n < N_b:
            n = N_b
        N_fft = 2**nextpow2(n)
    else:

        if N_x > N_b:
            N = 2**np.arange(np.ceil(np.log2(N_b)),np.floor(np.log2(N_x)))
            cost = np.ceil(N_x/(N-N_b+1))*N*(np.log2(N)+1)
            N_fft = N[np.argmin(cost)]
        else:
            N_fft = 2**nextpow2(N_b+N_x-1)

    N_fft = int(N_fft)
    
    # Compute the block length:
    L = int(N_fft - N_b + 1) 
    # Compute the transform of the filter:
    y = np.zeros((N_x,N_f),float)
    for v in range(N_f):
          H = np.fft.fft(b[:,v],N_fft)
          i = 0
          while i <= N_x:
                il = np.min([i+L,N_x])
                k = np.min([i+N_fft,N_x])
                yt = np.fft.ifft(np.fft.fft(x[i:il],N_fft)*H,N_fft) # Overlap..
                y[i:k,v] = y[i:k,v] + np.real(yt[:k-i])          # and add
                i += L
    return y




def get_GTfilters(nFilterBanks, samplerate, fRange, frameLen, overlap):
      
      filterOrder = 4
      gammaLen = 1024
      phase = np.zeros((nFilterBanks,))
    
      erb_b = hz2erb(np.array(fRange))
      erb = np.arange(erb_b[0], erb_b[1]+0.0001, np.diff(erb_b)/(nFilterBanks-1))
      centerFreq = erb2hz(erb)
      b = np.array(1.019*24.7*(4.37*centerFreq/1000+1))
      t_gammaTone = np.arange(1,gammaLen+1)/samplerate
      gammaToneFilters = np.zeros((gammaLen, nFilterBanks))
      for k in range(nFilterBanks):
            gain = (10**((loudness(centerFreq[k])-60)/20)/3)*(2*np.pi*b[k]/samplerate)**4
            gammaToneFilters[:,k] = gain*(samplerate**3)*(t_gammaTone**(filterOrder-1))*\
                                    np.exp(-2*np.pi*b[k]*t_gammaTone)*np.cos(2*np.pi*centerFreq[k]*t_gammaTone+phase[k])
      
      gamma_param = {}
      gamma_param['filterOrder'] = filterOrder;
      gamma_param['nFilterBanks'] = nFilterBanks;
      gamma_param['filtLen'] = gammaLen;
      gamma_param['cf'] = centerFreq;
      gamma_param['b'] = b;
      gamma_param['gFilters'] = gammaToneFilters;
      gamma_param['midEarCoeff'] = [10**((loudness(centerFreq[k])-60)/20) for k in range(len(centerFreq))]      
      gamma_param['fs'] = samplerate;
      gamma_param['framelen'] = int(np.round(frameLen*samplerate))
      gamma_param['overlap'] = int(np.round(overlap*samplerate))
      gamma_param['coswin'] = (1 + np.cos(2*np.pi*np.array(range(gamma_param['framelen']))/gamma_param['framelen'] - np.pi))/2;
      
      return gamma_param


def getFrames(signal, gamma_param):
      frameLen = gamma_param['framelen']
      overlap = gamma_param['overlap']
      numFrames = int(np.ceil((len(signal)-overlap)/overlap))
      signal2 = np.pad(signal,(0, (numFrames+1)*overlap - len(signal)), mode='constant')
      frameIdx = mb.repmat(np.arange(0,frameLen), numFrames, 1).transpose() + \
                 mb.repmat(np.arange(0,numFrames*overlap, overlap), frameLen, 1)
      frameData = signal2[frameIdx]*mb.repmat(np.sqrt(np.hamming(frameLen)), numFrames,1).transpose()
      return  frameData, frameIdx, numFrames

def softmax(input):
    regularization = 0.000001
    output = np.exp(input)/(np.sum(np.exp(input), axis=0)+regularization)
    output = (output - np.min(output))/(np.max(output)+regularization)
    return output



def zmean(input):
      output = (input - np.mean(input))/(np.std(input)+1e-3)
      output[np.isnan(output)] = 0
      return output.flatten()

def odd(f):
    f = int(np.ceil(f))
    return f + 1 if f % 2 == 0 else f


def GT_SPP_Enhance(x,gamma_param):
      gamma_mic = fftfilt(gamma_param['gFilters'], x)
      sklearn_pca = sklearnPCA(n_components=1,svd_solver='full')
      Mask = []
      
      for k in range(gamma_param['nFilterBanks']):
            gamma_frames, frameIdx, numFrames = getFrames(gamma_mic[:,k], gamma_param)
            
            # Energy
            Energy = zmean(20*np.log10(np.sum(np.abs(gamma_frames), axis=0)))
            
            # Peak2Valey Ratio
            PeakValey = zmean(20*np.log10(np.var(gamma_frames,axis=0)**2.1/ \
                              (np.var(np.abs(gamma_frames),axis=0)+1e-5)))
            
            # Entropy
            Entropy = np.zeros(numFrames)
            histBins = np.linspace(np.min(gamma_frames),np.max(gamma_frames), 30)
            for j in range(numFrames):
                  temp = np.histogram(np.abs(gamma_frames[:,j]),histBins)
                  prob = temp[0]/np.sum(temp[0]+1e-5)
                  Entropy[j] = 3.21 - np.sum((prob*np.log2(prob+1e-5))) 
            Entropy = zmean(Entropy)
            
            HigherOrderDiff = np.zeros(numFrames)
            diffFrames = 4
            for j in range(diffFrames):
                  HigherOrderDiff = HigherOrderDiff+ np.sum(np.abs(np.diff(gamma_frames, j+1, axis=0)),axis=0)
            HigherOrderDiff = zmean(HigherOrderDiff)
          
            Correlation = np.zeros(numFrames)
            corrCoeff = np.corrcoef(gamma_frames.transpose())
            corrFrames = 10
            for j in range(corrFrames,numFrames):
                  Correlation[j] = np.mean(corrCoeff[j,j-corrFrames:j])
            Correlation = zmean(Correlation) 
            
            featureMatrix = np.array([Energy, HigherOrderDiff, Correlation, PeakValey, Entropy])
            featureMatrix = sklearn_pca.fit_transform(featureMatrix.transpose()).flatten()
            featureMatrix = softmax(1.0/(1.0+np.exp(-1*featureMatrix)))
            
            Mask.append(featureMatrix)
            
      Mask = medfilt2d(Mask,(odd(gamma_param['nFilterBanks']/16),3))
      
      # Mask -> Speech Activity Detection (SAD)
      vad_temp = np.sum(Mask>0.3, axis=0)
      vad_temp = medfilt(vad_temp/np.max(np.abs(vad_temp)),11);
      vad_frames = vad_temp>0.8*np.mean(vad_temp)
      vad_samples = resample(vad_frames, len(x)) > 0.5;   
      
      
      # Reconstruction
      increment = gamma_param['framelen']/gamma_param['overlap']
      sigLength = len(x) #frameIdx[-1:,-1:][0][0]
      y = np.zeros((sigLength,))
      for k in range(gamma_param['nFilterBanks']):
            gamma_mic[:,k] = np.flipud(gamma_mic[:,k])/gamma_param['midEarCoeff'][k]
            temp = fftfilt_one(gamma_param['gFilters'][:,k],gamma_mic[:,k]);
            gamma_mic[:,k] = np.flipud(temp)/gamma_param['midEarCoeff'][k]
            
            weight = np.zeros((sigLength,));
            for m in range(numFrames-int(increment/2)+1):
                  startpoint = m*gamma_param['overlap']
                  if m <= int(increment/2):
                        weight[:startpoint+int(gamma_param['framelen']/2)-1] += Mask[k,m]*(gamma_param['coswin'][int(gamma_param['framelen']/2)-startpoint+1:])           
                  else:
                        weight[startpoint-int(gamma_param['framelen']/2):startpoint+int(gamma_param['framelen']/2)] += Mask[k,m]*gamma_param['coswin'] 
            
            y += gamma_mic[:,k]*weight; 
      y = np.max(np.abs(x))*y/np.max(np.abs(y))

      return y, vad_samples








