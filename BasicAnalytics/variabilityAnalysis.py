"""
Functions to analyse variability of movements
"""
import numpy as np

def calculateVariabilityScores(rigidBodyData,includePositions=False,positionsOnly = False):

    rotationIndicies = []

    # Add all suitable rotation indices so that array is [3,4,5,9,10,11....114]
    # Start at 3, increment by 6 for each pattern
    if includePositions:
        # Compute the derivative 
        derivative = centralDifference(rigidBodyData, points=5)
    
    else:
        if positionsOnly:
            startIdx = 0
        else:
            startIdx = 3
        for i in range(startIdx, 114, 6):  
            # Add three consecutive numbers 
            rotationIndicies.extend([i, i + 1, i + 2])

        # Compute the derivative 
        derivative = centralDifference(rigidBodyData[:,rotationIndicies,:], points=10)

    # Calculate RMS values
    rmsValues = calculateRmsForMultidimensionalFeatures(derivative)

    return rmsValues

def calculateVariabilityScoresInSegmentedTrial(rigidBodyData,includePositions = False):
    """
    This function takes in decoder data and gives off variability scores for each segment of each trial 
    """
    # Key sizes
    noTimes,noDOF,noDecoders,noParticipants = rigidBodyData.shape

    # How many segments to split variability metric into over trial
    noSegments = 5

    # Timestamps in  each segment
    segmentLength = noTimes // noSegments

    rotationIndicies = []

    # Add all suitable rotation indices so that array is [3,4,5,9,10,11....114]
    # Start at 3, increment by 6 for each pattern
    for i in range(3, 114, 6):  
        # Add three consecutive numbers 
        rotationIndicies.extend([i, i + 1, i + 2])

    

    # Hold output array 
    if includePositions:
        outputArr = np.zeros((noSegments,114,noDecoders,noParticipants))

    else:
        outputArr = np.zeros((noSegments,len(rotationIndicies),noDecoders,noParticipants))

    # Computer variability for all decoders
    for decoder in range(7):

        # Compute variability for all segments
        segmentStartingVals = range(0,noTimes,segmentLength)
        for segmentIdx in range(5):

            # Define start and end of segment
            start = segmentStartingVals[segmentIdx]
            end = start + segmentLength

            if includePositions:
                # Calculate derivative of segment for specific decoder
                derivative = centralDifference(rigidBodyData[start:end,:,decoder,:], points=5)
            else:
                # Calculate derivative of segment for specific decoder
                derivative = centralDifference(rigidBodyData[start:end,rotationIndicies,decoder,:], points=5)
            
            # Calculate RMS values of segment for specific decoder
            rmsValues = calculateRmsForMultidimensionalFeatures(derivative)

            outputArr[segmentIdx,:,decoder,:] = rmsValues

    return outputArr


def centralDifference(data, points=5):
    """
    Computes the derivative of an m x n x o array using the central differences formula.
    At the edges, use forward and backward differences to approximate the derivative.
    
    Inputs:
        data: A 3D numpy array of shape [m, n, o], m is number of time steps, and n is number of variables and o is number of trials
        points: Number of points on either side to use for the central difference.
    
    Returns:
        Numpy array of the derivatives with shape m x n x o
    """
    
    m, n, o = data.shape
    derivative = np.zeros((m, n, o))
    
    # Central difference for the bulk of the data
    for i in range(points, m - points):
        derivative[i, :] = (data[i + points, :,:] - data[i - points, :,:]) / (2 * points)
    
    # Forward difference for the beginning edge
    for i in range(points):
        derivative[i, :] = (data[i + points, :, :] - data[i, :, :]) / points

    # Backward difference for the ending edge
    for i in range(-points, 0):
        derivative[i, :] = (data[i, :, :] - data[i - points, :, :]) / points
        
    return derivative

def calculateRmsForMultidimensionalFeatures(data):
    """
    Calculates root mean square (RMS) for each feature in a 3D dataset.
    
    Args:
        data: A 3D numpy array of shape (m, n, o), where m is the number of observations,
            n is the number of features, and o is the dimensionality of each feature.
    
    Returns:
        2D numpy array of shape (n, o), containing the RMS value for each feature dimension.
    """
    # Square all values, take the mean over observations, 
    # then square root each mean to get the RMS.
    rms = np.sqrt(np.mean(data ** 2, axis=0))
    return rms
