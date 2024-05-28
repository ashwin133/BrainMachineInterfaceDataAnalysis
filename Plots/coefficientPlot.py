"""
Investigate linear model coefficients
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

# add current path to system PATH 
sys.path.insert(0,'/Users/ashwin/Documents/Y4 project Brain Human Interfaces/General 4th year Github repo/BrainMachineInterfaceDataAnalysis')

# Import user defined libraries
import DataExtraction.extractRawData as dataExtractor

# Fetch model coefficients
num2Let = ['E','F', 'G', 'H', 'I', 'J', 'K']
for i in range(6):
    letter = num2Let[i]
    coefficients, offsets, minsDOF, maxsDOF, DOFoffsets  = dataExtractor.extractDecoderWeights(decoder = letter)

    print(len(coefficients))
    plt.imshow(np.asarray(coefficients)[:,0,:])
    plt.imshow(np.asarray(coefficients)[:,1,:])
    plt.colorbar()
    plt.show()

# let's retrain using ridge decoders
for p
