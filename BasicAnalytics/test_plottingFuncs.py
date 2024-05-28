"""
tests for plotting functionalities
"""

import matplotlib.pyplot as plt
import numpy as np



def test_createBoxPlot():

    # Import user defined libraries
    import DataExtraction.extractRawData as dataExtractor
    from BasicAnalytics import targetAcqusitionPlotting as targetPlotter
    from BasicAnalytics import variabilityAnalysis 
    from BasicAnalytics import plottingFuncs


    # Fetch key training data for variability analysis
    rigidBodyTrain1, scores, noParticipants = dataExtractor.retrieveTrainingData()

    # Create a time array
    time_steps = np.linspace(0, 2*np.pi, 100)  # 100 time steps

    # Calculate variability values of training data
    rms_values = variabilityAnalysis.calculateVariabilityScores(rigidBodyTrain1)
    print(rms_values.shape)

    # Calculate variability values of individual body parts by summing over dof for each body part
    rmsValuesRigidBodyParts = np.sum(rms_values.reshape(19,3,noParticipants*5),axis = 1).reshape(19,noParticipants*5)
    print(rms_values)


    rms_values = rmsValuesRigidBodyParts

    # Score r values 
    maxVal = np.max(rms_values)
    rms_values = rms_values / maxVal

    # Normalisation and sum variability values across DOF
    summedRMSvaluesAcrossDOF = np.sum(rms_values.reshape(19,5,noParticipants),axis = 0)
    summedRMSvaluesAcrossDOF = summedRMSvaluesAcrossDOF / np.max(summedRMSvaluesAcrossDOF)

    red = (245/255,5/255,5/255) # (RGB) or F50505 (Hex)

    colors = [red, red, red, red, red]
    xTickList = ['Trial 1', 'Trial 2', 'Trial 3', 'Trial 4', 'Trial 5']
    trialSplitSummedRMSVals = [summedRMSvaluesAcrossDOF[0,:], summedRMSvaluesAcrossDOF[1,:], summedRMSvaluesAcrossDOF[2,:], summedRMSvaluesAcrossDOF[3,:],summedRMSvaluesAcrossDOF[4,:]]

    fig= plt.figure(figsize=(10,6))
    ax = plt.gca()
    plottingFuncs.createBoxPlot(ax, trialSplitSummedRMSVals, colors, xTickList, 'Trial No', 'Variability metric')
    plt.title("Box plot of variability of rigid bodies across trials \n with line of average variability", fontsize = 20)
    # Middle of box plot is median and line is average so there may be differences
    plt.plot(np.linspace(1,5,5),np.average(summedRMSvaluesAcrossDOF,axis = 1),label = "Average variability")
    plt.legend(fontsize = 15)
    fig.tight_layout() 