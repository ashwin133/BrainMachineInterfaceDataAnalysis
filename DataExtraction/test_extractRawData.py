"""
tests data extraction functionalities
"""

def test_processTrialData():
    # Import libraries
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import pickle
    import sys



    # add current path to system PATH 
    sys.path.insert(0,'/Users/ashwin/Documents/Y4 project Brain Human Interfaces/General 4th year Github repo/BrainMachineInterfaceDataAnalysis')

    # Import user defined libraries
    from DataExtraction.extractRawData import processTrialData
    # Define sample file to open
    location = "ExperimentRuns/P1_Saksham_20_02/P1_Saksham_20_02__11_25_usingDecoderG"

    # Extract all necessary data
    trialInformation = processTrialData(location)

    with open('DataExtraction/TestData/testProcessTrialData' + ".pkl","rb") as file:
        dict = pickle.load(file)

    assert np.array_equal(dict['rigidBodyData'],trialInformation['rigidBodyData'])

    assert all([ dict['timeInTargets'][a] == trialInformation['timeInTargets'][a] for a in range(len(trialInformation['timeInTargets'])) ])

    # Try decoder
    location = "ExperimentRuns/P1_Saksham_20_02/P1_Saksham_20_02__11_25_usingDecoderF"
    # Extract all necessary data
    trialInformation = processTrialData(location)


    # Try training
    location = "ExperimentRuns/P1_Saksham_20_02/P1_Saksham_20_02__11_25_training1"
    # Extract all necessary data
    trialInformation = processTrialData(location)

def test_extractSpecificParticipantFiles():

    # Import libraries
    import sys

    # add current path to system PATH 
    sys.path.insert(0,'/Users/ashwin/Documents/Y4 project Brain Human Interfaces/General 4th year Github repo/BrainMachineInterfaceDataAnalysis')

    # Import function
    from DataExtraction.extractRawData import extractSpecificParticipantFiles
    trainingFiles = extractSpecificParticipantFiles(trialType="_training1",dir = "ExperimentRuns")

    # Expected files function should find
    ans = ['ExperimentRuns/P1_Saksham_20_02/P1_Saksham_20_02__11_25_training1', 'ExperimentRuns/P2_Colin_20_02/P2_Colin_20_02__11_25_training1', 'ExperimentRuns/P3_Siddhi_21_02/P3_Siddhi_21_02__11_25_training1',
            'ExperimentRuns/P4_Hinze_28_02/P4_Hinze_28_02__12_00_training1', 'ExperimentRuns/P5_Katsu_28_02/P5_Katsu_28_02__12_00_training1', 'ExperimentRuns/P6_Tejas_28_02/P6_Tejas_28_02__12_00_training1', 
            'ExperimentRuns/P7_Pranjal_17_30/P7_Pranjal_29_02__17_30_training1', 'ExperimentRuns/P8_Saomiyan_04_03/P8_Saomiyan_04_03__16_54_training1', 'ExperimentRuns/P9_Bryant_05_03/P9_Bryant_05_03__12_08_training1', 
            'ExperimentRuns/P10_Adi_05_03/P10_Adi_05_03__17_08_training1']
    
    # Test function finds expected files
    assert trainingFiles == ans


def test_retrieveTrainingData():

    from DataExtraction.extractRawData import retrieveTrainingData

    rigidBodyTrain1, scores, noParticipants = retrieveTrainingData()


test_extractSpecificParticipantFiles()
