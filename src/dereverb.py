#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 15:48:46 2018

@author: vkk160330
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input.wav", help="Degraded audio file (absolute path)")
    parser.add_argument("output.wav", help="Enhanced audio file (absolute path)")
    parser.add_argument("-n", required=False, action='store_true', help="Number of Gammtone FilterBanks")
    parser.add_argument("-w", required=False, action='store_true', help="Window Length (ms)")
    parser.add_argument("-s", required=False, action='store_true', help="Overlap (ms)")

    args = parser.parse_args()
  