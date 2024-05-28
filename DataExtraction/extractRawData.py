"""
Contains functions to extract raw data from npz and pkl game files
"""



import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import sys
from tqdm import tqdm
from scipy.signal import butter, filtfilt

# add BMI path to system PATH as importing pkl files rely on game code
sys.path.insert(0,'/Users/ashwin/Documents/Y4 project Brain Human Interfaces/General 4th year Github repo/BrainMachineInterfaces')

def processTrialData(location,DOFOffset = 0.1,returnAsDict = True,scalefeaturesAndOutputs = True,timeLimit = 6000,endTime = 150000):
    """
    This function reads the raw trial data format that is generated after the pointer game is run. It processes 
    the data to extract useful information 
    INPUT:
    @param location: location of raw trial data. Ensure file is inside PointerExperimentData, only supply the file name not the .npz or .pkl bit
    @param DOFOffset: This adds the specified offset to each DOF's range , 0.1 used in games
    @param returnAsDict: return data in a dictionary or as variables
    @param scalefeaturesAndOutputs: decide whether to apply normative scaling to features and outputs
    @param timeLimit: time limit to hit targets in trial
    RETURNS:

    EITHER//
    @param rigidBodyData: all rigid body movements across whole trial (this has been calibrated)
    @param cursorMotion_noTimestamp: cursor motion across whole trial in form x,y
    @param cursorVelocities: cursor velocities across whole trial
    @param goCuesIdx: indexes into trial data corresponding to when target displayed on screen
    @param targetAquiredIdxes: index into trial data corresponding to when target reached
    @param timeStamps: timestamps for each index in trial data
    @param minDOF: minimum values for each DOF before scaling
    @param maxDOF: maximum values for each DOF before scaling

    OR//

    @param returnDict: a dict object of useful information for the trial

    """

    # Extract npz data for trial
    data = np.load(location + ".npz",allow_pickle=True) 
        
    
    # Extract pkl data for trial
    with open(location + ".pkl",'rb') as file:
        gameEngine,player = pickle.load(file)
    
    # Retrieve calibration data
    calibrationMatrix = player.calibrationMatrix

    # Retrieve score
    score = player.score

    # Use calibration matrix (c) (3x3) to construct 6x6 cal matrix in form [c,0;0,c]
    fullCalibrationMatrix = np.zeros((6,6))
    fullCalibrationMatrix[0:3,0:3] = calibrationMatrix
    fullCalibrationMatrix[3:6,3:6] = calibrationMatrix 


    # Fetch target Box positions
    targetBoxLocs = data['targetBoxLocs']

    # data starts as soon as cursor moves on screen
    # recieve list of cursor movements
    cursorMotion = data['cursorMotionDatastoreLocation']    
    
    # Calibrate rigid body data
    rigidBodyData = data['allBodyPartsData'] 
    rigidBodyData = rigidBodyData.reshape(-1,51,6)

    rawRigidBodyData = rigidBodyData.copy()

    # Apply calibration matrix to data
    rigidBodyData_normalised = np.tensordot(fullCalibrationMatrix,rigidBodyData.transpose(), axes=([1],[0])).transpose().reshape(-1,306)


    # Find when data stops being recorded for cursor data
    lastrecordCursorIdx = np.where(cursorMotion[:,0] == 0)[0][0] - 1
    lastrecordRigidBodyIdx = np.where(rigidBodyData_normalised[:,0] == 0)[0][0] - 1

    # Index is when calibration finishes and the cursor starts to move
    startRigidBodyIdx = lastrecordRigidBodyIdx - lastrecordCursorIdx 

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

    # Bug present so we need to delete first target
    targetBoxAppearTimes =  targetBoxAppearTimes[1:]
    targetBoxHitTimes = targetBoxHitTimes[1:]

    # Find successful target acquisitions
    successfulAcquires = [a != -1 for a in targetBoxHitTimes]

    # Find idxes of successful and failed target acquisitions
    successfulAcquiresIdxes = [i for i in range(len(targetBoxHitTimes)) if targetBoxHitTimes[i] != -1]
    failedAcquiresIdxes = [i for i in range(len(targetBoxHitTimes)) if targetBoxHitTimes[i] == -1]

    successfulTargetBoxAppearTimes = [targetBoxAppearTimes[i] for i in range(len(targetBoxHitTimes)) if targetBoxHitTimes[i] != -1]
    failedTargetBoxAppearTimes = [targetBoxAppearTimes[i] for i in range(len(targetBoxHitTimes)) if targetBoxHitTimes[i] == -1]

    # Now get rid of unsuccessful indicators, replace with a time of 10s after target
    successfulTargetBoxHitTimes =  [targetBoxHitTimes[i] for i in range(len(targetBoxHitTimes)) if targetBoxHitTimes[i] != -1]
    failedTargetBoxHitTimes = [targetBoxAppearTimes[i] + timeLimit for i in range(len(targetBoxHitTimes)) if targetBoxHitTimes[i] == -1]
    
    # Delete zero entries of box hit and appear times
    zeroIdx = np.where(targetBoxAppearTimes == 0)[0][0]
    targetBoxAppearTimes = targetBoxAppearTimes[0:zeroIdx]

    # Identify the index of timestamps for target appearing and hit
    succesfulGoCueIdxes = [np.argmin(np.abs(cursorMotion[:,0] - a)) for a in successfulTargetBoxAppearTimes]
    successfulTargetAquiredIdxes = [np.argmin(np.abs(cursorMotion[:,0] - a)) for a in successfulTargetBoxHitTimes]

    # Identify the index of timestamps for target appearing and user failing to hit
    failedGoCueIdxes = [np.argmin(np.abs(cursorMotion[:,0] - a)) for a in failedTargetBoxAppearTimes]
    failedTargetAquiredIdxes = [np.argmin(np.abs(cursorMotion[:,0] - a)) for a in failedTargetBoxHitTimes]

    # Remove timestamp column of cursor motion
    cursorMotion_noTimestamp = cursorMotion[:,1:] 

    # Delete zero entries of target box locs
    targetBoxLocs = targetBoxLocs[0:len(targetBoxAppearTimes),:]

    # Find successful and failed target box locs
    successfulTargetBoxLocs = [targetBoxLocs[a] for a in range(len(targetBoxHitTimes)) if targetBoxHitTimes[a] != -1]
    failedTargetBoxLocs = [targetBoxLocs[a] for a in range(len(targetBoxHitTimes)) if targetBoxHitTimes[a] == -1]
    
    # Normalise range of target boxes
    # boxWidth = 60
    # targetBoxLocs[:,0] += boxWidth // 2
    # targetBoxLocs[:,1] -= boxWidth // 2

    # targetBoxLocs[:,0] /= (1100 + 800)
    # targetBoxLocs[:,1] /= (800 + 225)

    timeStamps = cursorMotion[:,0]

    # Delete all redundant rigid bodies
    simpleBodyParts = [0,1,2,3,4,5,6,7,8,24,25,26,27,43,44,45,47,48,49]
    rigidBodyData  = rigidBodyData.reshape(-1,51,6)
    rigidBodyData = rigidBodyData[:,simpleBodyParts,:].reshape(-1,114)
    noDOF = 114
    maxDOF = np.zeros(114)
    minDOF = np.zeros(114)

    # If no sucesses then skip these steps
    if successfulTargetBoxHitTimes != []:
        # Find time user has been left each target
        timeLeftTarget = player.scoreUpdateTimes

        # delete first idx if this is 0
        if timeLeftTarget[0] == 0:
            del timeLeftTarget[0]
        
        if successfulTargetBoxHitTimes[-1] > timeLeftTarget[-1]:
            timeLeftTarget.append(endTime)

        if len(timeLeftTarget) - 1 == len(successfulTargetBoxHitTimes):
            del timeLeftTarget[0]
        
        

        # Find time user has remained in each target
        timeInTarget = [timeLeftTarget[a] - successfulTargetBoxHitTimes[a] for a in range(len(successfulTargetBoxHitTimes))]
    
    # Empty lists as no targets hit
    else:
        timeLeftTarget = []
        timeInTarget = []

    # Scale each rigid body and cursor if requested
    if scalefeaturesAndOutputs:
        for DOF in range(0,noDOF):
            DOFMin = min(rigidBodyData[:,DOF])
            minDOF[DOF] = DOFMin
            DOFMax = max(rigidBodyData[:,DOF])
            maxDOF[DOF] = DOFMax
            rigidBodyData[:,DOF] =  (rigidBodyData[:,DOF] - DOFMin) / (DOFMax - DOFMin + DOFOffset) # very sensitive to the offset ???

        # cursorDOF = 2s
        # for cursorDim in range(0,cursorDOF):
        #     cursorDOFmin = min(cursorMotion_noTimestamp[:,cursorDim])
        #     if True: # make min and max x,y cursor pos the actual range set in pygame
        #         if cursorDim == 0:
        #             cursorDOFmin = 0
        #             cursorDOFMax = 1100 + 800
        #         else:
        #             cursorDOFmin = 0
        #             cursorDOFmax = 800 + 225
        #     #cursorDOFmax = max(cursorMotion_noTimestamp[:,cursorDim])

        #     cursorMotion_noTimestamp[:,cursorDim] = (cursorMotion_noTimestamp[:,cursorDim] - cursorDOFmin) / (cursorDOFmax - cursorDOFmin+ 5)



    if returnAsDict is not True:
        return rigidBodyData, cursorMotion_noTimestamp,cursorVelocities,np.array(succesfulGoCueIdxes),np.array(successfulTargetAquiredIdxes), timeStamps,minDOF,maxDOF, successfulAcquires, successfulTargetBoxLocs
    else:
        returnDict = {
            'rigidBodyData': rigidBodyData,
            'cursorPos': cursorMotion_noTimestamp,
            'cursorVel': cursorVelocities,
            'successfulGoCues': np.array(succesfulGoCueIdxes),
            'successfulTargetReached': np.array(successfulTargetAquiredIdxes),
            'failedGoCues': np.array(failedGoCueIdxes),
            'failedTargetReached': np.array(failedTargetAquiredIdxes),
            'timestamps': timeStamps,
            'minDOF': minDOF,
            'maxDOF': maxDOF,
            'successfulAcquires': successfulAcquires,
            'successfulAcquireIdxes': successfulAcquiresIdxes,
            'failedAcquireIdxes': failedAcquiresIdxes,
            'successfulTargetBoxLocs': successfulTargetBoxLocs,
            'failedTargetBoxLocs': failedTargetBoxLocs,
            'timeInTargets': timeInTarget,
            'playerScore': score,
            'calibrationMatrix': fullCalibrationMatrix,
            'startRigidBodyIdx': startRigidBodyIdx,
            'rawRigidBodyData': rawRigidBodyData
        }
        return returnDict
    
import re
def natural_sort_key(s):
    """Sorting key to order strings by value."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
    
def extractSpecificParticipantFiles(trialType = "_training1", dir = "ExperimentRuns",useNpy = False,usePkl = False):
    """
    Function retrieves same trial game saves for all participants in the directory dir, and returns all files as a list

    Args:
        trialType: the end file identifier corresponding to the enquired trial
        dir: the directory containing sub directories to iterate over for each game save
    """
    # Base directory
    baseDirectory = dir

    # temporary add .pkl to end of trialType to find file
    if useNpy:
        trialType += ".npz"
    else:
        trialType += ".pkl"

    # List to hold the paths of the files
    trainingFiles = []

    

    # List sub directories
    dirnames = os.listdir(baseDirectory)

    # Sort the dirnames by numeric key
    dirnames =  sorted(dirnames, key=natural_sort_key)

    # Iterate through each subdir and find specific file
    for dirname in dirnames:

        # Look at files starting with P, ignores system hidden files
        if dirname.split(os.sep)[-1].startswith('P'):

            # List all files in subdir
            all_entries = os.listdir(os.path.join(baseDirectory,dirname))

            # Iterate through each file in subdir
            for filename in all_entries:

                # identify files that end with requested trial type
                if filename.endswith(trialType):

                    # Append the full path of the file to the list
                    fileName = os.path.join(baseDirectory,dirname,filename)

                    # Delete pkl part of filename
                    if useNpy:
                        pass
                        #fileName = fileName.replace(".npz", "") 
                    else:
                        if not usePkl:
                            fileName = fileName.replace(".pkl", "")
                    trainingFiles.append(fileName)

    return trainingFiles




def retrieveTrainingData():
    """
    Retrieves rigid body vector, scores and number of participants for all participants across all trials in training phase

    Returns:
        rigidBodyTrain1: rigid body vector of m x n x KP (m is number of timesteps, n is number of body degrees of freedom, P is participants and K is trials, trials increases slowest )
        scores: array of scores of shape P x K, where P is participant and K is trials
        noParticipants: number of participants
    """

    # Fetch participant files for training trial 1
    trainingFiles = extractSpecificParticipantFiles(trialType= "_training1")

    # Define no participants
    noParticipants = len(trainingFiles)

    # Define array to hold rigid body movements of participants
    rigidBodyTrain1 = np.zeros((7500, 114, 5*noParticipants)) 

    # Define array to hold scores of participants
    scores = np.zeros((noParticipants,5))


    # Collect trial 1 data
    print("Extracting trial 1 data ...")
    for idx in tqdm(range(len(trainingFiles))):

        file = trainingFiles[idx]

        # Extract data as dict from file
        trainData =  processTrialData(file)

        # Get rigid body data
        rigidBodyData = trainData['rigidBodyData']

        # Get score
        scores[idx,0] = trainData['playerScore']
        
        # store first 8000 samples
        rigidBodyTrain1[:,:,idx] = rigidBodyData[0:7500,:]

    # Fetch participant files for training trial 2
    trainingFiles = extractSpecificParticipantFiles(trialType="_training2")

    # Collect trial 2 data
    print("Extracting trial 2 data ...")
    for idx in tqdm(range(len(trainingFiles))):

        file = trainingFiles[idx]

        # Extract data as dict from file
        trainData =  processTrialData(file)

        # Get rigid body data
        rigidBodyData = trainData['rigidBodyData']

        # Get score
        scores[idx,1] = trainData['playerScore']
        
        # store first 8000 samples
        rigidBodyTrain1[:,:,noParticipants+idx] = rigidBodyData[0:7500,:]

    # Fetch participant files for training trial 3
    trainingFiles = extractSpecificParticipantFiles(trialType="_training3")

    # Collect trial 3 data
    print("Extracting trial 3 data ...")
    for idx in tqdm(range(len(trainingFiles))):

        file = trainingFiles[idx]

        # Extract data as dict from file
        trainData =  processTrialData(file)

        # Get rigid body data
        rigidBodyData = trainData['rigidBodyData']

        # Get score
        scores[idx,2] = trainData['playerScore']
        
        # store first 8000 samples
        rigidBodyTrain1[:,:,noParticipants*2+idx] = rigidBodyData[0:7500,:]

    # Fetch participant files for training trial 4
    trainingFiles = extractSpecificParticipantFiles(trialType="_training4")

    # Collect trial 4 data
    print("Extracting trial 4 data ...")
    for idx in tqdm(range(len(trainingFiles))):

        file = trainingFiles[idx]

        # Extract data as dict from file
        trainData = processTrialData(file)

        # Get rigid body data
        rigidBodyData = trainData['rigidBodyData']

        # Get score
        scores[idx,3] = trainData['playerScore']
        
        # store first 8000 samples
        rigidBodyTrain1[:,:,noParticipants*3+idx] = rigidBodyData[0:7500,:]

    # Fetch participant files for training trial 5
    trainingFiles = extractSpecificParticipantFiles(trialType="_test")

    # Collect trial 5 data
    print("Extracting trial 5 data ...")
    for idx in tqdm(range(len(trainingFiles))):

        file = trainingFiles[idx]

        # Extract data as dict from file
        trainData = processTrialData(file)

        # Get rigid body data
        rigidBodyData = trainData['rigidBodyData']

        # Get score
        scores[idx,4] = trainData['playerScore']
        
        # store first 8000 samples
        rigidBodyTrain1[:,:,noParticipants*4+idx] = rigidBodyData[0:7500,:]

    return rigidBodyTrain1, scores, noParticipants


def retrieveDecoderData(version = 1):
    """
    Retrieves rigid body vector, scores and number of participants for all participants across all trials in decoder phase

    Returns:
        Version = 1:
            rigidBodyData: rigid body vector of m x n x K x P (m is number of timesteps, n is number of body degrees of freedom, P is participants and K is decoder trials, trials increases slowest )
            scores: array of scores of shape P x K, where P is participant and K is trials
            noParticipants: number of participants
        
        Version = 2:
            rigidBodyData: rigid body vector of m x n x K x P (m is number of timesteps, n is number of body degrees of freedom, P is participants and K is decoder trials, trials increases slowest )
            positionList: positions of cursor position of shape m x D x K x P (m is number of timesteps, D is DOF (x,y), P is participants and K is decoder trials, trials increases slowest )
            scores: array of scores of shape P x K, where P is participant and K is trials
            noParticipants: number of participants
            trialInformationLists: list of dicts containing trial information about each trial (length is PK), listed by all participants in first decoder , then all participants in second decoder and etc.

    """

    # Set correct path
    os.chdir('/Users/ashwin/Documents/Y4 project Brain Human Interfaces/General 4th year Github repo/BrainMachineInterfaceDataAnalysis')
    # Fetch participant files for training trial E
    decoderFiles = extractSpecificParticipantFiles(trialType= "_usingDecoderE")

    # Define no participants
    noParticipants = len(decoderFiles)

    # Define array to hold rigid body movements of participants
    rigidBodyDecoderData = np.zeros((5589, 114, 7,noParticipants)) 

    # Define array to hold timestamps
    timeStampData = np.zeros((5589, 7,noParticipants)) 

    # Define array to hold scores of participants
    scores = np.zeros((noParticipants,7))

    # Define array to hold positions of cursors
    positionList = np.zeros((5589, 2, 7,noParticipants)) 

    # Define list to hold dicts of information for each trial list is in [Decoder E all participants, Decoder F all participants etc.]
    trialInformationLists = []

    # Collect decoder E data
    print("Extracting Decoder E data ...")
    for idx in tqdm(range(len(decoderFiles))):

        file = decoderFiles[idx]

        # Extract data as dict from file
        trainData =  processTrialData(file)

        # Get rigid body data
        rigidBodyData = trainData['rigidBodyData']

        # Get score
        scores[idx,0] = trainData['playerScore']

        # Get positions
        positions = trainData['cursorPos']

        # Get times
        timestamps = trainData['timestamps']

        # Store data
        timeStampData[:,0,idx] = timestamps[0:5589]

        # store first 8000 samples
        rigidBodyDecoderData[:,:,0,idx] = rigidBodyData[0:5589,:]

        # Store positions
        positionList[:,:,0,idx] = positions[0:5589,:]

        # Extract all necessary data
        trialInformationLists.append(trainData)

    # Fetch participant files for decoder F
    decoderFiles = extractSpecificParticipantFiles(trialType="_usingDecoderF")

    # Collect decoder F data
    print("Extracting Decoder F data ...")
    for idx in tqdm(range(len(decoderFiles))):

        file = decoderFiles[idx]

        # Extract data as dict from file
        trainData =  processTrialData(file)

        # Get rigid body data
        rigidBodyData = trainData['rigidBodyData']

        # Get score
        scores[idx,1] = trainData['playerScore']

        # Get times
        timestamps = trainData['timestamps']

        # Store data
        timeStampData[:,1,idx] = timestamps[0:5589]
        
        # store first 8000 samples
        rigidBodyDecoderData[:,:,1,idx] = rigidBodyData[0:5589,:]

        # Get positions
        positions = trainData['cursorPos']

        # Store positions
        positionList[:,:,1,idx] = positions[0:5589,:]

        # Extract all necessary data
        trialInformationLists.append(trainData)

    # Fetch participant files for decoder G
    decoderFiles = extractSpecificParticipantFiles(trialType="_usingDecoderG")

    # Collect decoder G data
    print("Extracting decoder G data ...")
    for idx in tqdm(range(len(decoderFiles))):

        file = decoderFiles[idx]

        # Extract data as dict from file
        trainData =  processTrialData(file)

        # Get rigid body data
        rigidBodyData = trainData['rigidBodyData']

        # Get score
        scores[idx,2] = trainData['playerScore']

        # Get times
        timestamps = trainData['timestamps']

        # Store data
        timeStampData[:,2,idx] = timestamps[0:5589]
        
        # store first 8000 samples
        rigidBodyDecoderData[:,:,2,idx] = rigidBodyData[0:5589,:]

        # Get positions
        positions = trainData['cursorPos']

        # Store positions
        positionList[:,:,2,idx] = positions[0:5589,:]

        # Extract all necessary data
        trialInformationLists.append(trainData)

    # Fetch participant files for decoder H
    decoderFiles = extractSpecificParticipantFiles(trialType="_usingDecoderH")

    # Collect decoder H data
    print("Extracting decoder H data ...")
    for idx in tqdm(range(len(decoderFiles))):

        file = decoderFiles[idx]

        # Extract data as dict from file
        trainData = processTrialData(file)

        # Get rigid body data
        rigidBodyData = trainData['rigidBodyData']

        # Get score
        scores[idx,3] = trainData['playerScore']

        # Get times
        timestamps = trainData['timestamps']

        # Store data
        timeStampData[:,3,idx] = timestamps[0:5589]
        
        # store first 8000 samples
        rigidBodyDecoderData[:,:,3,idx] = rigidBodyData[0:5589,:]

        # Get positions
        positions = trainData['cursorPos']

        # Store positions
        positionList[:,:,3,idx] = positions[0:5589,:]

        # Extract all necessary data
        trialInformationLists.append(trainData)

    # Fetch participant files for decoder I
    decoderFiles = extractSpecificParticipantFiles(trialType="_usingDecoderI")

    # Collect decoder I data
    print("Extracting decoder I data ...")
    for idx in tqdm(range(len(decoderFiles))):

        file = decoderFiles[idx]

        # Extract data as dict from file
        trainData = processTrialData(file)

        # Get rigid body data
        rigidBodyData = trainData['rigidBodyData']

        # Get score
        scores[idx,4] = trainData['playerScore']

        # Get times
        timestamps = trainData['timestamps']

        # Store data
        timeStampData[:,4,idx] = timestamps[0:5589]
        
        # store first 8000 samples
        rigidBodyDecoderData[:,:,4,idx] = rigidBodyData[0:5589,:]

        # Get positions
        positions = trainData['cursorPos']

        # Store positions
        positionList[:,:,4,idx] = positions[0:5589,:]

        # Extract all necessary data
        trialInformationLists.append(trainData)
    
    # Fetch participant files for decoder J
    decoderFiles = extractSpecificParticipantFiles(trialType="_usingDecoderJ")

    # Collect decoder J data
    print("Extracting decoder J data ...")
    for idx in tqdm(range(len(decoderFiles))):

        file = decoderFiles[idx]

        # Extract data as dict from file
        trainData = processTrialData(file)

        # Get rigid body data
        rigidBodyData = trainData['rigidBodyData']

        # Get score
        scores[idx,5] = trainData['playerScore']

        # Get times
        timestamps = trainData['timestamps']

        # Store data
        timeStampData[:,5,idx] = timestamps[0:5589]
        
        # store first 8000 samples
        rigidBodyDecoderData[:,:,5,idx] = rigidBodyData[0:5589,:]

        # Get positions
        positions = trainData['cursorPos']

        # Store positions
        positionList[:,:,5,idx] = positions[0:5589,:]

        # Extract all necessary data
        trialInformationLists.append(trainData)
    
    # Fetch participant files for decoder K
    decoderFiles = extractSpecificParticipantFiles(trialType="_usingDecoderK")

    # Collect decoder K data
    print("Extracting decoder k data ...")
    for idx in tqdm(range(len(decoderFiles))):

        file = decoderFiles[idx]

        # Extract data as dict from file
        trainData = processTrialData(file)

        # Get rigid body data
        rigidBodyData = trainData['rigidBodyData']

        # Get score
        scores[idx,6] = trainData['playerScore']

        # Get times
        timestamps = trainData['timestamps']

        
        
        # Get positions
        positions = trainData['cursorPos']
        
        # store first 8000 samples
        try:
            rigidBodyDecoderData[:,:,6,idx] = rigidBodyData[0:5589,:]


            # Store data
            timeStampData[:,6,idx] = timestamps[0:5589]
            # Store positions
            positionList[:,:,6,idx] = positions[0:5589,:]

            # Extract all necessary data
            trialInformationLists.append(trainData)
        except:
            length = len(rigidBodyData[:,0])
            print("length 1",length)
            rigidBodyDecoderData[:length,:,6,idx] = rigidBodyData[:,:]
            length = min(len(positions[:,0]),5589)

            # Store data
            timeStampData[:length,6,idx] = timestamps[:length]
            print("length 2",length)
            positionList[:length,:,6,idx] = positions[:length,:]
            trialInformationLists.append(trainData)

    if version == 1:
        return rigidBodyDecoderData, scores, noParticipants,  
    elif version == 2:
        return rigidBodyDecoderData, positionList, scores, noParticipants,  trialInformationLists
    elif version == 3:
        return rigidBodyDecoderData, scores, noParticipants,  timeStampData

def extractDecoderWeights(decoder = "G"):
    """
    For a particular decoder this function extracts the decoder weights for all participants

    returns:
        coefficients : list of coefficients for rigid bodies for all participants
        offsets: list of offsets for all participants
        minsDOF: list of min DOF values for all participants
        maxsDOF: list of max DOF values for all participants
        DOFoffsets: list of DOF offset values for all participants
    """

     # Set correct path
    os.chdir('/Users/ashwin/Documents/Y4 project Brain Human Interfaces/General 4th year Github repo/BrainMachineInterfaceDataAnalysis')
    # Fetch participant files for decoder
    decoderFiles = extractSpecificParticipantFiles(trialType= "_linearRigidBody{}Model".format(decoder),useNpy=True)

    # Define no participants
    noParticipants = len(decoderFiles)

    # Retrieve decoder mappings for each participant
    
    # Store each participants coefficients in a list
    coefficients = []

    # Store each participants offsets in a list
    offsets = []

    # Store each participants min and max DOF motion in a list
    minsDOF = []
    maxsDOF = []

    # Store DOF offset 
    DOFoffsets = []

    for idx in tqdm(range(len(decoderFiles))):
        
        file = decoderFiles[idx]

        # Load npz file
        data = np.load(file)

        # Add coefficients to list
        coefficients.append(data['modelCoeff'])

        # Add offset to list
        offsets.append(data['modelIntercept'])

        # Add min and max possible movements to list
        minsDOF.append(data['minDOF'])
        maxsDOF.append(data['maxDOF'])

        DOFoffsets.append(data['DOFOffset'])

    # Return vars
    return coefficients, offsets, minsDOF, maxsDOF, DOFoffsets