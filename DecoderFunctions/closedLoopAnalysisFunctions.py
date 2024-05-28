"""
Contains some helper functions to post process the raw data so models can be applied

"""
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

def processTrialData(dataLocation,calLocation,DOFOffset = 0.03,returnAsDict = False,scalefeaturesAndOutputs = True,ignoreCalibration = False):
    """
    This function reads the raw trial data format that is generated after the pointer game is run. It processes 
    the data to extract useful information to fit models
    INPUT:
    @param dataLocation: location of raw trial data. Ensure file is inside PointerExperimentData, only supply the file name
    @param calLocation: location of file which has calibration matrix stored.  Ensure file is inside PointerExperimentData, only supply the file name
    # supply none to fetch the calibration matrix from the game engine
    @param DOFOffset: This adds the specified offset to each DOF's range
    @param returnAsDict: return data in a dictionary or as variables
    RETURNS:
    @param rigidBodyData: all rigid body movements across whole trial
    @param cursorMotion_noTimestamp: cursor motion across whole trial in form x,y
    @param cursorVelocities: cursor velocities across whole trial
    @param goCuesIdx: indexes into trial data corresponding to when target displayed on screen
    @param targetAquiredIdxes: index into trial data corresponding to when target reached
    @param timeStamps: timestamps for each index in trial data
    @param minDOF: minimum values for each DOF before scaling
    @param maxDOF: maximum values for each DOF before scaling
    """

    try:

        data = np.load( dataLocation,allow_pickle=True) # for siddhi trial 3 the boxes were 60 x 60
        
    except FileNotFoundError:

        data = np.load('Experiment_pointer/PointerExperimentData/' + dataLocation,allow_pickle=True)
    
    if calLocation != None:
        # Fetch the calibration matrix from a file
        try:
            calMatrix = np.load('../PointerExperimentData/' + calLocation)
        except:
            calMatrix = np.load('Experiment_pointer/PointerExperimentData/' + calLocation)

        # recieve list of transformed rigid body vectors that correspond to cursor movements
        calMatrix = calMatrix['calMatrix']

    else:
        # Fetch the calibration matrix from the game engine
        if ignoreCalibration == False:
            gameEngine = data['gameEngineLocation']
            calMatrix = gameEngine.fullCalibrationMatrix
        

    # Fetch target Box positions
    targetBoxLocs = data['targetBoxLocs']


    # data starts as soon as cursor moves on screen
    # recieve list of cursor movements
    cursorMotion = data['cursorMotionDatastoreLocation']    
    
    # Calibrate rigid body data
    rigidBodyData_trial1 = data['allBodyPartsData'] # raw motion of all rigid bodies
    rigidBodyData_trial1 = rigidBodyData_trial1.reshape(-1,51,6)
    if ignoreCalibration == False:
        rigidBodyData_normalised = np.tensordot(calMatrix,rigidBodyData_trial1.transpose(), axes=([1],[0])).transpose().reshape(-1,306)
    else:
        rigidBodyData_normalised = rigidBodyData_trial1.reshape((-1,306))
    

    # Find when data stops being recorded for cursor data
    lastrecordCursorIdx = np.where(cursorMotion[:,0] == 0)[0][0] - 1
    lastrecordRigidBodyIdx = np.where(rigidBodyData_normalised[:,0] == 0)[0][0] - 1
    startRigidBodyIdx = lastrecordRigidBodyIdx - lastrecordCursorIdx # as this is when calibration finishes and the cursor starts to move

    # Start rigid body data after calibration
    rigidBodyData = rigidBodyData_normalised[startRigidBodyIdx:lastrecordRigidBodyIdx+1,:]
    
    # Stop cursor motion data after last non zero value, as initially it is an array of size 0 larger than needed
    cursorMotion = cursorMotion[0:lastrecordCursorIdx+1]
    
    # Calculate cursor velocities 
    cursorVelocities = np.gradient(cursorMotion[:,1:],cursorMotion[:,0],axis=0)

    # Zero any erroneous values (values wildly above reasonable limits in both columns)
    # cursorMotion[:,1] = [cursorMotion[a,1] if abs(cursorMotion[a,1]) < 3000 else 0 for a in range(cursorMotion[:,1].shape[0])]
    # cursorMotion[:,2] = [cursorMotion[a,2] if abs(cursorMotion[a,2]) < 3000 else 0 for a in range(cursorMotion[:,2].shape[0])]

    # now get times of when target appeared to when target was hit
    targetBoxHitTimes = np.array(data['targetBoxHitTimes'])
    targetBoxAppearTimes = np.array(data['targetBoxAppearTimes'])
    # get the relevant elements of targetBoxAppearTimes

    # Find successful target acquisitions
    successfulAcquires = [a != -1 for a in targetBoxHitTimes  ]

    # Now get rid of unsuccessful indicators, replace with a time of 10s after target
    targetBoxHitTimes = [targetBoxHitTimes[i] if targetBoxHitTimes[i]  != -1 else targetBoxAppearTimes[i] + 10000 for i in range(len(targetBoxHitTimes)) ]

    # Delete zero entries of box hit and appear times
    zeroIdx = np.where(targetBoxAppearTimes == 0)[0][0]
    targetBoxAppearTimes = targetBoxAppearTimes[0:zeroIdx]

    # Identify the index of timestamps for target appearing and hit
    goCueIdxes = [np.argmin(np.abs(cursorMotion[:,0] - a)) for a in targetBoxAppearTimes]
    targetAquiredIdxes = [np.argmin(np.abs(cursorMotion[:,0] - a)) for a in targetBoxHitTimes]

    # Remove timestamp column of cursor motion
    cursorMotion_noTimestamp = cursorMotion[:,1:] 

    # Delete zero entries of target box locs
    targetBoxLocs = targetBoxLocs[0:len(goCueIdxes),:]

    # Normalise range of target boxes
    # boxWidth = 60
    # targetBoxLocs[:,0] += boxWidth // 2
    # targetBoxLocs[:,1] -= boxWidth // 2

    targetBoxLocs[:,0] /= 1100 + 800
    targetBoxLocs[:,1] /= 800 + 225

    timeStamps = cursorMotion[:,0]

    # Delete all redundant rigid bodies
    simpleBodyParts = [0,1,2,3,4,5,6,7,8,24,25,26,27,43,44,45,47,48,49]
    rigidBodyData  = rigidBodyData.reshape(-1,51,6)
    rigidBodyData = rigidBodyData[:,simpleBodyParts,:].reshape(-1,114)
    noDOF = 114
    maxDOF = np.zeros(114)
    minDOF = np.zeros(114)

    if len(goCueIdxes) == len(targetAquiredIdxes) + 1:
        del goCueIdxes[-1]
        targetBoxLocs = targetBoxLocs[1:,:]

    # Bug with one participant so manually take care of it here
    if dataLocation == "ExperimentRuns/P6_Tejas_28_02/P6_Tejas_28_02__12_00_usingDecoderF.npz":
        del goCueIdxes[0]
        del targetAquiredIdxes[0]
        targetBoxLocs = targetBoxLocs[:-1,:]
        del successfulAcquires[0]

    # Scale each rigid body and cursor if requested

    if scalefeaturesAndOutputs:
        for DOF in range(0,noDOF):
            DOFMin = min(rigidBodyData[:,DOF])
            minDOF[DOF] = DOFMin
            DOFMax = max(rigidBodyData[:,DOF])
            maxDOF[DOF] = DOFMax
            rigidBodyData[:,DOF] =  (rigidBodyData[:,DOF] - DOFMin) / (DOFMax - DOFMin + DOFOffset) # very sensitive to the offset ???

        cursorDOF = 2
        for cursorDim in range(0,cursorDOF):
            cursorDOFmin = min(cursorMotion_noTimestamp[:,cursorDim])
            if True: # make min and max x,y cursor pos the actual range set in pygame
                if cursorDim == 0:
                    cursorDOFmin = 0
                    cursorDOFmax = 1100 + 800
                else:
                    cursorDOFmin = 0
                    cursorDOFmax = 800 + 225
            #cursorDOFmax = max(cursorMotion_noTimestamp[:,cursorDim])

            cursorMotion_noTimestamp[:,cursorDim] = (cursorMotion_noTimestamp[:,cursorDim] - cursorDOFmin) / (cursorDOFmax - cursorDOFmin)



    if returnAsDict is not True:
        return rigidBodyData, cursorMotion_noTimestamp,cursorVelocities,np.array(goCueIdxes),np.array(targetAquiredIdxes), timeStamps,minDOF,maxDOF, successfulAcquires, targetBoxLocs
    else:
        returnDict = {
            'rigidBodyData': rigidBodyData,
            'cursorPos': cursorMotion_noTimestamp,
            'cursorVel': cursorVelocities,
            'goCues': np.array(goCueIdxes),
            'targetReached': np.array(targetAquiredIdxes),
            'timestamps': timeStamps,
            'minDOF': minDOF,
            'maxDOF': maxDOF,
            'successfulAcquires': successfulAcquires,
            'targetBoxLocs': targetBoxLocs
        }
        return returnDict
    
def processMulipleTrialData(dataLocations,calLocation, DOFOffset =0.03):
    """
    Collates multiple trial data information together
    """

    trialDict = readIndividualTargetMovements(processTrialData(dataLocations[0],calLocation,DOFOffset,returnAsDict=True))
    del dataLocations[0]
    for file in dataLocations:
        trialDict_ = readIndividualTargetMovements(processTrialData(file,calLocation,DOFOffset,returnAsDict=True))
        for key in trialDict.keys():

            trialDict[key] = trialDict_[key] + trialDict[key]
    return trialDict
        
def readIndividualTargetMovements(processedDataDict):
    returnDict = {
        'rigidBodyData': [], 
        'cursorPosData': [], 
        'cursorVelData': [],
        'timestamps': []
    }
    for i in range(len(processedDataDict['goCues']) - 1):

        startTime = processedDataDict['timestamps'][processedDataDict['goCues'][i]]
        returnDict['rigidBodyData'].append(processedDataDict['rigidBodyData'][processedDataDict['goCues'][i]:processedDataDict['targetReached'][i],:].transpose().copy())
        returnDict['cursorPosData'].append(processedDataDict['cursorPos'][processedDataDict['goCues'][i]:processedDataDict['targetReached'][i],:].transpose().copy())
        returnDict['cursorVelData'].append(processedDataDict['cursorVel'][processedDataDict['goCues'][i]:processedDataDict['targetReached'][i],:].transpose().copy())
        returnDict['timestamps'].append((processedDataDict['timestamps'][processedDataDict['goCues'][i]:processedDataDict['targetReached'][i]] - startTime).copy())    
        
    return returnDict

def plotVar(var1,list_ = False,invertY = False,npArray = False,plotFrom = 0,plotTo = -1,label = "",var1Label = "true",points = False):
    colorMap =  [
    'red',         # Standard named color
    '#FFA07A',     # Light Salmon (hexadecimal)
    'blue',        # Standard named color
    '#00FA9A',     # Medium Spring Green (hexadecimal)
    'green',       # Standard named color
    '#FFD700',     # Gold (hexadecimal)
    'purple',      # Standard named color
    '#87CEFA',     # Light Sky Blue (hexadecimal)
    'orange',      # Standard named color
    '#FF69B4',     # Hot Pink (hexadecimal)
    'cyan',        # Standard named color
    '#8A2BE2',     # Blue Violet (hexadecimal)
    'magenta',     # Standard named color
    '#20B2AA',     # Light Sea Green (hexadecimal)
    'brown',       # Standard named color
    '#D2691E',     # Chocolate (hexadecimal)
    'pink',        # Standard named color
    '#6495ED'      # Cornflower Blue (hexadecimal)

    ]

    if points:
        # Plot all points from a list of 2d points

        for idx,var in enumerate(var1):
            if idx == 0:
                # For label
                plt.plot(var[0],var[1],marker = 'X',markersize = 10,color = colorMap[idx%len(colorMap)],label = "Target Location")
            else:
                plt.plot(var[0],var[1],marker = 'X',markersize = 10,color = colorMap[idx%len(colorMap)])

    elif list_:
        for idx,var in enumerate(var1):
            if npArray == False:
                var = np.asarray(var).transpose()
            if invertY:
                var[plotFrom:plotTo,1] = 1 - var[plotFrom:plotTo,1]

            if len(var) <= 70 and idx != 0:
                continue

            if idx == 0:
                plt.plot(var[plotFrom:plotTo,0],var[plotFrom:plotTo,1],color = "k",label = "Trajectory")
                plt.plot(var[plotFrom,0],var[plotFrom,1],marker = '.',markersize = 20,color = 'k',label = "Start")
                plt.plot(var[plotTo-1,0],var[plotTo-1,1],marker = 's',markersize = 10, color = 'k', label = "End")
            
            else:
                plt.plot(var[plotFrom:plotTo,0],var[plotFrom:plotTo,1],color = colorMap[idx%len(colorMap)])
                plt.plot(var[plotFrom,0],var[plotFrom,1],marker = '.',markersize = 20,color = colorMap[idx%len(colorMap)])
                plt.plot(var[plotTo-1,0],var[plotTo-1,1],marker = 's',markersize = 10, color = colorMap[idx%len(colorMap)])
            
    else:

        plt.plot(var1[plotFrom:plotTo,0],var1[plotFrom:plotTo,1],label = var1Label +label,color = colorMap[idx%len(colorMap)])
        plt.plot(var1[plotFrom,0],var1[plotFrom,1],marker = '.',markersize = 20,color = colorMap[idx%len(colorMap)])
        plt.plot(var1[plotTo,0],var1[plotTo,1],marker = 'x',markersize = 10, color = colorMap[idx%len(colorMap)])
    plt.legend(loc = "upper right",bbox_to_anchor=(1, 1),fontsize = 15)

def calcNormalisedAcquisitionTimes(processedDataDict,start = 0, end = -1,reactionTime = 300):
    """
    This function takes in data giving the start and location for each acquisition and 
    """
    cursorPosData = processedDataDict['cursorPosData'][start:end]
    timestamps = processedDataDict['timestamps'][start:end]
    IPs = []


    for idx,var in enumerate(cursorPosData):
        var = var.transpose()
        # Calculate the distance to each target from the starting point
        startPos = (var[0,0],var[0,1])
        endPos = (var[-1,0],var[-1,1])
        distToTarget = np.sqrt(np.sum([ (endPos[i] - startPos[i]) ** 2 for i in range(len(startPos))]))
        
        # Calculate distance unnormalised distance for fitts law
        ranges = [1100+800,800+225]
        D = np.sqrt(np.sum([ (ranges[i] * (endPos[i] - startPos[i])) ** 2 for i in range(len(startPos))]))
        if D == 0:
            continue
        # Calculate the time difference  
        timeStart = timestamps[idx][0]
        timeEnd = timestamps[idx][-1]




        # the raw acquisition time before being normalised for distance and reaction time
        totalTime = timeEnd - timeStart
        cursorWidth = 60
        IP = calcHumanPerformance(totalTime,reactionTime,D,W=60)
        IPs.append(IP)
        

        
    plt.plot(IPs, label = "Index of Performances")
    plt.show()
    return IPs

def calcHumanPerformance(totalTime,reactionTime,D,W):
    """uses Fitts law to calculate human performance"""
    MT = totalTime - reactionTime
    ID = np.log2((2*D)/W)
    IP = ID / MT
    return IP

def calcNormalisedAcquisitionTimes_old(processedDataDict,start = 0, end = -1,reactionTime = 300):
    """
    This function takes in data giving the start and location for each acquisition and 
    """
    cursorPosData = processedDataDict['cursorPosData'][start:end]
    timestamps = processedDataDict['timestamps'][start:end]
    acquisitionTimes = []


    for idx,var in enumerate(cursorPosData):
        var = var.transpose()
        # Calculate the distance to each target from the starting point
        startPos = (var[0,0],var[0,1])
        endPos = (var[-1,0],var[-1,1])
        distToTarget = np.sqrt(np.sum([ (endPos[i] - startPos[i]) ** 2 for i in range(len(startPos))]))
        # Calculate distance unnormalised distance for fitts law
        ranges = [1100+800,800+225]
        distToTarget_unNormalised = np.sqrt(np.sum([ (ranges[i] * (endPos[i] - startPos[i])) ** 2 for i in range(len(startPos))]))
        print(distToTarget_unNormalised)
        # Calculate the time difference  
        timeStart = timestamps[idx][0]
        timeEnd = timestamps[idx][-1]




        # the raw acquisition time before being normalised for distance and reaction time
        rawAcquisitionTime = timeEnd - timeStart

        acquisitionTime = (rawAcquisitionTime - reactionTime) / distToTarget

        acquisitionTimes.append(acquisitionTime)

        

        
    plt.plot(acquisitionTimes, label = "Normalised Acquisition Time")
    plt.show()
    return acquisitionTimes
    

