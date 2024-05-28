"""
Tests for functions analysing variability of movements
"""
import numpy as np

def test_calculateVariabilityScores():

    import DataExtraction.extractRawData as dataExtractor
    from BasicAnalytics import targetAcqusitionPlotting as targetPlotter
    from BasicAnalytics import variabilityAnalysis 



    rigidBodyTrain1, scores, noParticipants = dataExtractor.retrieveTrainingData()


    # Create a time array
    time_steps = np.linspace(0, 2*np.pi, 100)  # 100 time steps

    rms_values = variabilityAnalysis.calculateVariabilityScores(rigidBodyTrain1)