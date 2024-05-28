"""
tests files for targetAcquisitionPlotting
"""

def test_plotFailedTrajectoryToTarget():

    # Import libraries
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import pickle
    import sys

    # add current path to system PATH 
    sys.path.insert(0,'/Users/ashwin/Documents/Y4 project Brain Human Interfaces/General 4th year Github repo/BrainMachineInterfaceDataAnalysis')

    # Import user defined libraries
    import DataExtraction.extractRawData as dataExtractor
    from BasicAnalytics import targetAcqusitionPlotting as targetPlotter

    # Define file to open
    location = "ExperimentRuns/P1_Saksham_20_02/P1_Saksham_20_02__11_25_usingDecoderG"

    # Extract all necessary data
    trialInformation = dataExtractor.processTrialData(location)

    # Plot failed trajectory 
    targetPlotter.plotFailedTrajectoryToTarget(trialInformation,0)


    #plt.show()
