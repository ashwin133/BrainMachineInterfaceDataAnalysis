"""
functions used in experimental flow
"""

# Import libraries

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import sys
import pandas as pd
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm

# add current path to system PATH 
sys.path.insert(0,'/Users/ashwin/Documents/Y4 project Brain Human Interfaces/General 4th year Github repo/BrainMachineInterfaceDataAnalysis')

# Import user defined libraries
import DataExtraction.extractRawData as dataExtractor
from BasicAnalytics import targetAcqusitionPlotting as targetPlotter
from BasicAnalytics import variabilityAnalysis 
from BasicAnalytics import plottingFuncs as pf
from tol_colors import tol_cmap, tol_cset
cmap = tol_cset('vibrant')
from DecoderFunctions import closedLoopAnalysisFunctions as decoderCL

import numpy as np
import os 
import sys
import pickle

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import NotebookExporter

def find_indices_of_minus_ones(arr):
    indices = []
    for i, val in enumerate(arr):
        if val == -1:
            indices.append(i)
    return indices




"""# THE NEXT FUNCTIONS ARE USED TO CREATE DECODERS"""



"""
This script calculates for a trial, the average direction error for each target aquisition 

"""
# function
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn.decomposition import PCA
from sklearn import linear_model
sys.path.insert(0,'/Users/ashwin/Documents/Y4 project Brain Human Interfaces/General 4th year Github repo/BrainMachineInterfaceDataAnalysis')

from DecoderFunctions.config_streaming import renderingBodyParts,simpleBodyParts

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


def calcThetha(y_curr,y_prev,x_curr,x_prev):
    dY = y_curr - y_prev
    dX = x_curr - x_prev
    if dY == 0 and dX == 0:
        #print('zero detected')
        raise(ZeroDivisionError)
    
    if dX == 0: # Avoid zero division error
        dX = 0.00001
    alpha = np.abs(np.arctan((y_curr - y_prev)/(dX)))
    if dY >= 0 and dX >= 0:
        return alpha
    elif dY >= 0 and dX <= 0:
        return np.pi - alpha
    elif dY <= 0 and dX >= 0:
        return 2 * np.pi -alpha
    elif dY <= 0 and dX <= 0:
        return  np.pi + alpha
    
def calcMag(y_curr,y_prev,x_curr,x_prev):
    dY = y_curr - y_prev
    dX = x_curr - x_prev
    return np.sqrt(np.square(dY) + np.square(dX))


def calcDirAvgForMovingToTarget(cursorPos,timeStamps):
    thethas = []
    for i in range(5,len(cursorPos[:,0]),5):
        try:
            thethas.append((calcThetha(cursorPos[i,1],cursorPos[i-5,1],cursorPos[i,0],cursorPos[i-5,0])) * (timeStamps[i]-timeStamps[i-5]))
        except ZeroDivisionError:
            pass
    return (sum(thethas))/(timeStamps[-1] - timeStamps[0])

def calcMagAvgForMovingToTarget(cursorPos,timeStamps):
    mags = []
    for i in range(5,len(cursorPos[:,0]),5):
        try:
            mags.append((calcMag(cursorPos[i,1],cursorPos[i-5,1],cursorPos[i,0],cursorPos[i-5,0])) * (timeStamps[i]-timeStamps[i-5]))
        except ZeroDivisionError:
            pass
    return (sum(mags))/(timeStamps[-1] - timeStamps[0])

def reportThethaErrorForEachTargetMove(targetMotionCursorPosTrue,targetMotionCursorPosEst,targetMotionTimeStamps):
    thethaAvgTrue = calcDirAvgForMovingToTarget(targetMotionCursorPosTrue,targetMotionTimeStamps)
    thethaAvgEst = calcDirAvgForMovingToTarget(targetMotionCursorPosEst,targetMotionTimeStamps)
    return thethaAvgTrue, thethaAvgEst, thethaAvgTrue - thethaAvgEst

def reportMagErrorForEachTargetMove(targetMotionCursorPosTrue,targetMotionCursorPosEst,targetMotionTimeStamps):
    magAvgTrue = calcMagAvgForMovingToTarget(targetMotionCursorPosTrue,targetMotionTimeStamps)
    magAvgEst = calcMagAvgForMovingToTarget(targetMotionCursorPosEst,targetMotionTimeStamps)
    return magAvgTrue, magAvgEst, magAvgTrue - magAvgEst
def feedTargetMotionCursorPos(trueTrialCursorPos,estTrialCursorPos,goCueIdxes,targetAquiredIdxes,timeStamps,ignoreTargetMotionTImesLessThan):
    trialDirDifferences = np.zeros(len(targetAquiredIdxes))
    anglesTrue = []
    anglesEst = []
    times = []

    trialMagDifferences = np.zeros(len(targetAquiredIdxes))
    magsTrue = []
    magsEst = []
    times_mags = []
    for i in range(len(targetAquiredIdxes)):
        cursorPosTargetMotionTrue = trueTrialCursorPos[goCueIdxes[i]: targetAquiredIdxes[i],:]
        cursorPosTargetMotionEst = estTrialCursorPos[goCueIdxes[i]: targetAquiredIdxes[i], :]
        targetMotionTimestamps = timeStamps[goCueIdxes[i]: targetAquiredIdxes[i]]
        if targetMotionTimestamps[-1] - targetMotionTimestamps[0] > ignoreTargetMotionTImesLessThan:
            times.append(targetMotionTimestamps[-1] - targetMotionTimestamps[0])
            thetaTrue,thetaEst, thetaDiff = reportThethaErrorForEachTargetMove(cursorPosTargetMotionTrue,cursorPosTargetMotionEst,targetMotionTimestamps)
            magTrue,magEst,magDiff = reportMagErrorForEachTargetMove(cursorPosTargetMotionTrue,cursorPosTargetMotionEst,targetMotionTimestamps)
            trialDirDifferences[i] = (thetaDiff)
            anglesTrue.append(thetaTrue)
            anglesEst.append(thetaEst)
            thetaTrueDeg = np.rad2deg(thetaTrue)
            thetaEstDeg = np.rad2deg(thetaEst)
            thetaDiffDeg = np.rad2deg(thetaDiff)

            trialMagDifferences[i] = (magDiff)
            magsTrue.append(magTrue)
            magsEst.append(magEst)

            if False:
                plt.plot(cursorPosTargetMotionTrue[:,0],cursorPosTargetMotionTrue[:,1],marker = 'o')
                plt.plot(cursorPosTargetMotionEst[:,0],cursorPosTargetMotionEst[:,1],marker = 'o')
                plt.scatter(cursorPosTargetMotionTrue[0,0],cursorPosTargetMotionTrue[0,1],s=200, marker="D", color = 'g', label = 'True cursor trajectory start')
                plt.scatter(cursorPosTargetMotionEst[0,0],cursorPosTargetMotionEst[0,1],s=200, marker="D", color = 'r', label = 'Estimated cursor trajectory start')
                plt.title(f'Trajectory shown for a target aquisition task \n True average angle {thetaTrueDeg:.1f} deg, Estimated average angle {thetaEstDeg:.1f} deg, Average angular error {thetaDiffDeg:.1f} deg',fontsize = 15)
                plt.xlabel('Normalised X cursor position', fontsize = 15)
                plt.ylabel('Normalised Y cursor position', fontsize = 15)
                plt.legend()
                plt.show()
    return anglesTrue,anglesEst,trialDirDifferences, times, magsTrue,magsEst,trialMagDifferences


def processTrialData(dataLocation,DOFOffset = 0.03):
    """
    RETURNS:
    @param rigidBodyData: all rigid body movements across whole trial, this has been calibrated
    @param cursorMotion_noTimestamp: cursor motion across whole trial in form x,y
    @param cursorVelocities: cursor velocities across whole trial
    @param goCuesIdx: indexes into trial data corresponding to when target displayed on screen
    @param targetAquiredIdxes: index into trial data corresponding to when target reached
    @param timeStamps: timestamps for each index in trial data
    @param minDOF: minimum values for each DOF before scaling
    @param maxDOF: maximum values for each DOF before scaling
    """
    #print(os.getcwd())
    try:
        data = np.load('../' + dataLocation + ".npz") # 

    except FileNotFoundError:
        try:
            data = np.load(dataLocation + ".npz")
        except FileNotFoundError:
            dataLocation = dataLocation.replace("Experiment_pointer","..")
            
            data = np.load(dataLocation + ".npz")
        
    with open(dataLocation + ".pkl",'rb') as file:
        gameEngine,player = pickle.load(file)

    calMatrix_ = player.calibrationMatrix
    calMatrix = np.zeros((6,6))
    calMatrix[0:3,0:3] = calMatrix_
    calMatrix[3:6,3:6] = calMatrix_
    # data starts as soon as cursor moves on screen
    # recieve list of cursor movements
    cursorMotion = data['cursorMotionDatastoreLocation']    
    cursorExp = cursorMotion[:,1:]
    # recieve list of transformed rigid body vectors that correspond to cursor movements
    #print(calMatrix)
    rigidBodyData_trial1 = data['allBodyPartsData'] # raw motion of all rigid bodies
    rigidBodyData_trial1 = rigidBodyData_trial1.reshape(-1,51,6)
    rigidBodyData_normalised = np.tensordot(calMatrix,rigidBodyData_trial1.transpose(), axes=([1],[0])).transpose().reshape(-1,306)
    #print(rigidBodyData_normalised)

    

    # find when data stops being recorded for cursor data
    lastrecordCursorIdx = np.where(cursorMotion[:,0] == 0)[0][0] - 1
    lastrecordRigidBodyIdx = np.where(rigidBodyData_normalised[:,0] == 0)[0][0] - 1
    startRigidBodyIdx = lastrecordRigidBodyIdx - lastrecordCursorIdx # as this is when calibration finishes and the cursor starts to move

    rigidBodyData = rigidBodyData_normalised[startRigidBodyIdx:lastrecordRigidBodyIdx+1,:]
    
    cursorMotion = cursorMotion[0:lastrecordCursorIdx+1]
    
    cursorVelocities = np.gradient(cursorMotion[:,1:],cursorMotion[:,0],axis=0)

    # now get times of when target appeared to when target was hit
    targetBoxHitTimes = np.array(data['targetBoxHitTimes'])
    targetBoxAppearTimes = np.array(data['targetBoxAppearTimes'])
    
    # Identify which indexes are target misses
    targetMisses = find_indices_of_minus_ones(targetBoxHitTimes)

    # Create new array
    targetMissAppearTimes = targetBoxAppearTimes[targetMisses]

    boxAppearTimes = []
    boxHitTimes = []
    for i in range(len(targetBoxHitTimes)-1):
        if targetBoxHitTimes[i] == -1:
            pass
        else:
            boxHitTimes.append(targetBoxHitTimes[i])
            boxAppearTimes.append(targetBoxAppearTimes[i])

    targetBoxAppearTimes = np.array(boxAppearTimes)
    targetBoxHitTimes = np.array(boxHitTimes)

    # for i in targetMisses:
    #     targetBoxHitTimes[i] = targetBoxAppearTimes + gameEngine.timeLimit

    
    # get the relevant elements of targetBoxAppearTimes
    # zeroIdx = np.where(targetBoxAppearTimes == 0)[0][0]
    # targetBoxAppearTimes = targetBoxAppearTimes[0:zeroIdx]
    goCueIdxes = [np.argmin(np.abs(cursorMotion[:,0] - a)) for a in targetBoxAppearTimes]
    targetAquiredIdxes = [np.argmin(np.abs(cursorMotion[:,0] - a)) for a in targetBoxHitTimes]

    cursorMotion_noTimestamp = cursorMotion[:,1:] # remove timestamp column
    timeStamps = cursorMotion[:,0]
    simpleBodyParts = [0,1,2,3,4,5,6,7,8,24,25,26,27,43,44,45,47,48,49]
    rigidBodyData  = rigidBodyData.reshape(-1,51,6)
    rigidBodyData = rigidBodyData[:,simpleBodyParts,:].reshape(-1,114)
    noDOF = 114
    maxDOF = np.zeros(114)
    minDOF = np.zeros(114)

    if True:
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
                    cursorDOFMax = 1100 + 800
                else:
                    cursorDOFmin = 0
                    cursorDOFmax = 800 + 225
            #TODO: CHANGE THIS AND SEE HOW THIS AFFECTS THIS 
            cursorDOFmax = max(cursorMotion_noTimestamp[:,cursorDim])

            cursorMotion_noTimestamp[:,cursorDim] = (cursorMotion_noTimestamp[:,cursorDim] - cursorDOFmin) / (cursorDOFmax - cursorDOFmin)

    # def plotCursorMotion(cursorMotion):
    #     ax = plt.figure()
    #     plt.plot(cursorMotion[0:400,1],-cursorMotion[0:400,2])
    #     plt.show()
    # #plotCursorMotion(cursorMotion)

    
    return rigidBodyData, cursorMotion_noTimestamp,cursorVelocities,goCueIdxes,targetAquiredIdxes, timeStamps,minDOF,maxDOF,cursorExp, targetMissAppearTimes

def fitModelToData(mode,tester,compPca,savePath,colorMap = None,plot = False,DOFOffset = 0.03,ignoreTargetMotionTimesLessThan = 600,analysis = False,kFolds = None,participantNo = 0):
    if analysis:
        path = "/Users/ashwin/Documents/Y4 project Brain Human Interfaces/General 4th year Github repo/BrainMachineInterfaces/Experiment_pointer/DataAnalysis"
        os.chdir(path)
    rigidBodies1, cursorPos1,cursorVel1,goCues1,targetHits1,timeStamps1, minDof1,maxDof1,c,targetMissAppearTimes1 = processTrialData(savePath + "_test",DOFOffset)# make this test 
    rigidBodies2, cursorPos2,cursorVel2,goCues2,targetHits2,timeStamps2, minDof2,maxDof2,d,targetMissAppearTimes2 = processTrialData(savePath + "_training1",DOFOffset)
    rigidBodies3, cursorPos3,cursorVel3,goCues3,targetHits3,timeStamps3, minDof3,maxDof3,e,targetMissAppearTimes3 = processTrialData(savePath + "_training2",DOFOffset)
    rigidBodies4, cursorPos4,cursorVel4,goCues4,targetHits4,timeStamps4, minDof4,maxDof4,f,targetMissAppearTimes4 = processTrialData(savePath + "_training3",DOFOffset)
    rigidBodies5, cursorPos5,cursorVel5,goCues5,targetHits5,timeStamps5, minDof5,maxDof5,g,targetMissAppearTimes5 = processTrialData(savePath + "_training4",DOFOffset)

    # This needs to be updated for k folds
    if kFolds is not None:
        if kFolds == 1:
            rigidBodyVectorTraining = np.concatenate((rigidBodies2,rigidBodies3,rigidBodies4,rigidBodies5), axis = 0)
            cursorPosTraining = np.concatenate((cursorPos2,cursorPos3,cursorPos4,cursorPos5),axis = 0)
            maxDofTraining = np.max(np.concatenate((maxDof2,maxDof3,maxDof4,maxDof5)).reshape(-1,4),1)
            minDofTraining = np.min(np.concatenate((minDof2,minDof3,minDof4,minDof5)).reshape(-1,4),1)

            rigidBodyVectorTest = rigidBodies1
            cursorPosTest = cursorPos1
            goCuesTest = goCues1
            targetHitsTest = targetHits1
            timeStampsTest = timeStamps1

        elif kFolds == 2:
            rigidBodyVectorTraining = np.concatenate((rigidBodies3,rigidBodies4,rigidBodies5,rigidBodies1), axis = 0)
            cursorPosTraining = np.concatenate((cursorPos3,cursorPos4,cursorPos5,cursorPos1),axis = 0)
            maxDofTraining = np.max(np.concatenate((maxDof3,maxDof4,maxDof5,maxDof1)).reshape(-1,4),1)
            minDofTraining = np.min(np.concatenate((minDof3,minDof4,minDof5,minDof1)).reshape(-1,4),1)

            rigidBodyVectorTest = rigidBodies2
            cursorPosTest = cursorPos2
            goCuesTest = goCues2
            targetHitsTest = targetHits2
            timeStampsTest = timeStamps2
        
        elif kFolds == 3:
            rigidBodyVectorTraining = np.concatenate((rigidBodies4,rigidBodies5,rigidBodies1,rigidBodies2), axis = 0)
            cursorPosTraining = np.concatenate((cursorPos4,cursorPos5,cursorPos1,cursorPos2),axis = 0)
            maxDofTraining = np.max(np.concatenate((maxDof4,maxDof5,maxDof1,maxDof2)).reshape(-1,4),1)
            minDofTraining = np.min(np.concatenate((minDof4,minDof5,minDof2,minDof2)).reshape(-1,4),1)

            rigidBodyVectorTest = rigidBodies3
            cursorPosTest = cursorPos3
            goCuesTest = goCues3
            targetHitsTest = targetHits3
            timeStampsTest = timeStamps3
        
        elif kFolds == 4:
            rigidBodyVectorTraining = np.concatenate((rigidBodies5,rigidBodies1,rigidBodies2,rigidBodies3), axis = 0)
            cursorPosTraining = np.concatenate((cursorPos5,cursorPos1,cursorPos2,cursorPos3),axis = 0)
            maxDofTraining = np.max(np.concatenate((maxDof5,maxDof1,maxDof2,maxDof3)).reshape(-1,4),1)
            minDofTraining = np.min(np.concatenate((minDof5,minDof1,minDof2,minDof3)).reshape(-1,4),1)

            rigidBodyVectorTest = rigidBodies4
            cursorPosTest = cursorPos4
            goCuesTest = goCues4
            targetHitsTest = targetHits4
            timeStampsTest = timeStamps4

        elif kFolds == 5:
            rigidBodyVectorTraining = np.concatenate((rigidBodies1,rigidBodies2,rigidBodies3,rigidBodies4), axis = 0)
            cursorPosTraining = np.concatenate((cursorPos1,cursorPos2,cursorPos3,cursorPos4),axis = 0)
            maxDofTraining = np.max(np.concatenate((maxDof1,maxDof2,maxDof3,maxDof4)).reshape(-1,4),1)
            minDofTraining = np.min(np.concatenate((minDof1,minDof2,minDof3,minDof4)).reshape(-1,4),1)

            rigidBodyVectorTest = rigidBodies5
            cursorPosTest = cursorPos5
            goCuesTest = goCues5
            targetHitsTest = targetHits5
            timeStampsTest = timeStamps5

    else:

        rigidBodyVectorTraining = np.concatenate((rigidBodies2,rigidBodies3,rigidBodies4,rigidBodies5), axis = 0)
        cursorPosTraining = np.concatenate((cursorPos2,cursorPos3,cursorPos4,cursorPos5),axis = 0)
        maxDofTraining = np.max(np.concatenate((maxDof2,maxDof3,maxDof4,maxDof5)).reshape(-1,4),1)
        minDofTraining = np.min(np.concatenate((minDof2,minDof3,minDof4,minDof5)).reshape(-1,4),1)

        rigidBodyVectorTest = rigidBodies1
        cursorPosTest = cursorPos1
        goCuesTest = goCues1
        targetHitsTest = targetHits1
        timeStampsTest = timeStamps1

    if mode == 'RigidBodiesSetA':
        # delete right hand only SET A
        type = 'A'
        idxRightHand = renderingBodyParts.index('RHand') * 6
        X_train = np.delete(rigidBodyVectorTraining,slice(idxRightHand,idxRightHand+6,1),1)
        X_test_linear = np.delete(rigidBodyVectorTest,slice(idxRightHand,idxRightHand+6,1),1)
        X_test_ridge = np.delete(rigidBodyVectorTest,slice(idxRightHand,idxRightHand+6,1),1)

    elif mode == 'RigidBodiesSetB':
        # delete right side rigid bodies, SET B
        type = 'B'
        idxRightHand = renderingBodyParts.index('RHand') * 6
        idxRightShoulder = renderingBodyParts.index('RShoulder') * 6
        X_train = np.delete(rigidBodyVectorTraining,slice(idxRightShoulder,idxRightHand+6,1),1)
        X_test_linear = np.delete(rigidBodyVectorTest,slice(idxRightShoulder,idxRightHand+6,1),1)
        X_test_ridge = np.delete(rigidBodyVectorTest,slice(idxRightShoulder,idxRightHand+6,1),1)
    
    elif mode == 'RigidBodiesSetC':
        # # only get the left hand
        type = 'C'
        idxLeftHand = renderingBodyParts.index('LHand') * 6
        X_train = rigidBodyVectorTraining[:,idxLeftHand:idxLeftHand+6]
        X_test_linear = rigidBodyVectorTest[:,idxLeftHand:idxLeftHand+6]
        X_test_ridge = rigidBodyVectorTest[:,idxLeftHand:idxLeftHand+6]
    
    elif mode == 'RigidBodiesSetD':
        # # # only get the right hand
        type = 'D'
        idxRightHand = renderingBodyParts.index('RHand') * 6
        X_train = rigidBodyVectorTraining[:,idxRightHand:idxRightHand+6]
        X_test_linear = rigidBodyVectorTest[:,idxRightHand:idxRightHand+6]
        X_test_ridge = rigidBodyVectorTest[:,idxRightHand:idxRightHand+6]
    
    elif mode == 'RigidBodiesSetE':
        # # # angles only, use all non right hand
        type = 'E'
        # Find index of right hand on principal skeleton
        idxRightHand = renderingBodyParts.index('RHand') * 3
        rigidBodyVectorTraining = rigidBodyVectorTraining.reshape(-1,19,6)[:,:,3:].reshape(-1,19*3)
        rigidBodyVectorTest = rigidBodyVectorTest.reshape(-1,19,6)[:,:,3:].reshape(-1,19*3)
        
        X_train = np.delete(rigidBodyVectorTraining,slice(idxRightHand,idxRightHand+3,1),1)
        X_test_linear = np.delete(rigidBodyVectorTest,slice(idxRightHand,idxRightHand+3,1),1)
        X_test_ridge = np.delete(rigidBodyVectorTest,slice(idxRightHand,idxRightHand+3,1),1)
        
    
    elif mode == 'RigidBodiesSetF':
        # # # angles only, use all except right side
        type = 'F'

        # Find index of right hand on principal skeleton
        idxRightHand = renderingBodyParts.index('RHand') * 3

        # Find index of right shoulder on principal skeleton
        idxRightShoulder = renderingBodyParts.index('RShoulder') * 3

        # Retrieve all rigid body rotations for all timestamps
        rigidBodyVectorTraining = rigidBodyVectorTraining.reshape(-1,19,6)[:,:,3:].reshape(-1,19*3)
        rigidBodyVectorTest = rigidBodyVectorTest.reshape(-1,19,6)[:,:,3:].reshape(-1,19*3)
        
        # Delete rigid bodies on the right side
        X_train = np.delete(rigidBodyVectorTraining,slice(idxRightShoulder,idxRightHand+3,1),1)
        X_test_linear = np.delete(rigidBodyVectorTest,slice(idxRightShoulder,idxRightHand+3,1),1)
        X_test_ridge = np.delete(rigidBodyVectorTest,slice(idxRightShoulder,idxRightHand+3,1),1)
    

    elif mode == 'RigidBodiesSetG':
        # # # Angles only: only get the left hand
        type = 'G'
        # Find index of left hand in principal rigid bodies
        idxLeftHand = renderingBodyParts.index('LHand') * 6

        # Extract only the left hand rotations for test and training sets
        X_train = rigidBodyVectorTraining[:,idxLeftHand+3:idxLeftHand+6]
        X_test_linear = rigidBodyVectorTest[:,idxLeftHand+3:idxLeftHand+6]
        X_test_ridge = rigidBodyVectorTest[:,idxLeftHand+3:idxLeftHand+6]
    
    elif mode == 'RigidBodiesSetH':
        # # # only get the right hand
        type = 'H'

        # Find index of right hand in principal rigid bodies
        idxRightHand = renderingBodyParts.index('RHand') * 6

        # Extract only right hand rotations for test and training sets
        X_train = rigidBodyVectorTraining[:,idxRightHand+3:idxRightHand+6]
        X_test_linear = rigidBodyVectorTest[:,idxRightHand+3:idxRightHand+6]
        X_test_ridge = rigidBodyVectorTest[:,idxRightHand+3:idxRightHand+6]

    elif mode == 'RigidBodiesSetI':
        # # # Only use the lower body
         # # # angles only
        type = 'I'

        # Find index of left thigh on principal skeleton as this is start of lower bodies
        idxLeftThigh = renderingBodyParts.index('LThigh') * 3

        # Find index of right foot on principal skeleton as this is end of lower bodies
        idxRightFoot = renderingBodyParts.index('RFoot') * 3
        

        # Retrieve all rigid body rotations for all timestamps
        rigidBodyVectorTraining = rigidBodyVectorTraining.reshape(-1,19,6)[:,:,3:].reshape(-1,19*3)
        rigidBodyVectorTest = rigidBodyVectorTest.reshape(-1,19,6)[:,:,3:].reshape(-1,19*3)
        
        # Extract lower rigid bodies only
        X_train = rigidBodyVectorTraining[:,idxLeftThigh:idxRightFoot+3]
        X_test_linear = rigidBodyVectorTest[:,idxLeftThigh:idxRightFoot+3]
        X_test_ridge = rigidBodyVectorTest[:,idxLeftThigh:idxRightFoot+3]

    
    elif mode == 'RigidBodiesSetJ':
        # # # Only use the upper body ( excludes upper left and right)
         # # # angles only
        type = 'J'

        # Find index of neck on principal skeleton as this is start of upper body
        idxNeck = renderingBodyParts.index('Neck') * 3

        # Find index of head on principal skeleton as this is end of upper body
        idxHead = renderingBodyParts.index('Head') * 3
        

        # Retrieve all rigid body rotations for all timestamps
        rigidBodyVectorTraining = rigidBodyVectorTraining.reshape(-1,19,6)[:,:,3:].reshape(-1,19*3)
        rigidBodyVectorTest = rigidBodyVectorTest.reshape(-1,19,6)[:,:,3:].reshape(-1,19*3)
        
        # Extract upper rigid bodies only
        X_train = rigidBodyVectorTraining[:,idxNeck:idxHead+3]
        X_test_linear = rigidBodyVectorTest[:,idxNeck:idxHead+3]
        X_test_ridge = rigidBodyVectorTest[:,idxNeck:idxHead+3]

    elif mode == "RigidBodiesSetK":
        # Initialise random weights
        type = 'k'

        # Just take all data and develop model, then randomise coefficients
        X_train = rigidBodyVectorTraining.reshape(-1,19,6)[:,:,3:].reshape(-1,19*3)
        X_test_linear = rigidBodyVectorTest.reshape(-1,19,6)[:,:,3:].reshape(-1,19*3)




    if tester == 'PCA_linear':
        pca = PCA(n_components=compPca)
        pca.fit(X_train)
        pcaRes = pca.explained_variance_ratio_
        np.set_printoptions(suppress=True)
        #print(pcaRes)
        components = pca.components_
        #print(pca.components_)
        cumSumVar = np.cumsum(pcaRes)
        # plt.plot(cumSumVar)
        # plt.show()
        # transform all matrices to lower dimension
        X_train_pca = np.matmul(components,X_train.transpose()).transpose()
        Y_train = cursorPosTraining
        reg  = linear_model.LinearRegression().fit(X_train_pca, Y_train)
        # predict for trial 1
        X_test_linear_pca = np.matmul(components,X_test_linear.transpose()).transpose()
        Y_test_linear = reg.predict(X_test_linear_pca)
        score = reg.score(X_test_linear_pca,cursorPosTest)
        #print('Score:' ,score)
        Y_pred = Y_test_linear
        Y_test_true = cursorPosTest
    
    elif tester == 'linear':
        Y_train = cursorPosTraining
        reg  = linear_model.LinearRegression().fit(X_train, Y_train)
        
        # if model is k then randomise:
        if type == 'k':

            # For reproducibility
            # This should be participant inde
            np.random.seed(11)
            

            # Generate random weights
            random_coeff = np.random.normal(size = reg.coef_.shape)
            random_intercept = np.random.normal(size = reg.intercept_.shape)

            # Assign weights to the model
            reg.coef_ = random_coeff
            reg.intercept_ = random_intercept

            #print("Random Coefficients: ", reg.coef_)
            #print("Random Intercept: ", reg.intercept_)

        # predict for trial 1
        Y_test_linear = reg.predict(X_test_linear)
        score = reg.score(X_test_linear,cursorPosTest)
        #print('Score:' , score)
        Y_pred = Y_test_linear
        Y_test_true = cursorPosTest

    elif tester == 'ridge':
        Y_train = cursorPosTraining
        reg  = linear_model.Ridge().fit(X_train, Y_train)
        # predict for trial 1
        

        if type == 'k':

            # For reproducibility
            np.random.seed(participantNo)
            

            # Generate random weights
            random_coeff = np.random.normal(size = reg.coef_.shape)
            random_intercept = np.random.normal(size = reg.intercept_.shape)

            # Assign weights to the model
            reg.coef_ = random_coeff
            reg.intercept_ = random_intercept

            #print("Random Coefficients: ", reg.coef_)
            #print("Random Intercept: ", reg.intercept_)

        Y_test_linear = reg.predict(X_test_linear)
        score = reg.score(X_test_linear,cursorPosTest)
        print('Score:' , score)
        Y_pred = Y_test_linear
        Y_test_true = cursorPosTest


    if plot:
        correctY = cursorPosTest
        for i in range(0,len(colorMap)): # len(goCuesTest)
            plotFrom = goCuesTest[i]
            plotUntil = targetHitsTest[i]
            plt.plot(correctY[plotFrom:plotUntil,0],correctY[plotFrom:plotUntil,1],color = colorMap[i])
            plt.plot(Y_pred[plotFrom:plotUntil,0],Y_pred[plotFrom:plotUntil,1], color = colorMap[i])
            if i == 0:
                plt.scatter(correctY[plotFrom,0], correctY[plotFrom,1],s=250, marker=".", color = 'g',label = 'Actual cursor start position')
                # plt.scatter(correctY[plotUntil,0], correctY[plotUntil,1], s=100, marker="D", color = 'g')
                plt.scatter(Y_pred[plotFrom,0], Y_pred[plotFrom,1],s=250, marker=".", color = 'r',label = 'Estimated cursor start position')
                # plt.scatter(Y_pred[plotUntil,0], Y_pred[plotUntil,1], s=60, marker="D", color = 'r')
            else:
                plt.scatter(correctY[plotFrom,0], correctY[plotFrom,1],s=250, marker=".", color = 'g')
                # plt.scatter(correctY[plotUntil,0], correctY[plotUntil,1], s=100, marker="D", color = 'g')
                plt.scatter(Y_pred[plotFrom,0], Y_pred[plotFrom,1],s=250, marker=".", color = 'r')
                # plt.scatter(Y_pred[plotUntil,0], Y_pred[plotUntil,1], s=60, marker="D", color = 'r')
        
        plt.xlabel('Normalised X pos on game screen',fontsize = 15)
        plt.ylabel('Normalised Y pos on game screen', fontsize = 15)
        plt.title('Trajectories showing actual and estimated cursor position for each target aquisition performed in test set. \n Each trajectory is shown in a different colour and position estimates are derived from set (b) of rigid bodies',fontsize = 15)
        plt.legend()
        plt.show()
    if compPca is not None: 
        scoreLabel =  'l_PCA:' +str(compPca) + ', ' + str(DOFOffset)
    else:
        scoreLabel = 'l' +  ', ' + str(DOFOffset)

    # now calculate average angular error on the test dataset
    if goCuesTest[0] == targetHitsTest[0]:
        del goCuesTest[0]
        del targetHitsTest[0]
    trueAngles,estAngles, angularErrors,times,magsTrue,magsEst,trialMagDifferences = feedTargetMotionCursorPos(Y_test_true,Y_pred,goCuesTest,targetHitsTest,timeStampsTest,ignoreTargetMotionTimesLessThan)
    percentErrors = 100 * [abs((abs(estAngles[i] - trueAngles[i]))/trueAngles[i]) for i in range(0,len(trueAngles))]
    avgPercentError = np.average((percentErrors)) 
    percentErrorsMag = 100 * [abs(abs((magsEst[i] - magsTrue[i]))/magsTrue[i]) for i in range(0,len(magsTrue))]
    avgPercentErrorMag = np.average((percentErrorsMag)) 
    #print('error', avgPercentError)
    angularAccuracyMetric = 1 - np.abs(avgPercentError) / 1
    magAccuracyMetric = 1 - np.abs(avgPercentErrorMag) /1
    if tester == 'PCA_linear':
        label = type + ', l_PCA'  + str(compPca) + ', ' + str(DOFOffset) + ',<' + str(ignoreTargetMotionTimesLessThan) 
        modelCoeff = reg.coef_
        modelIntercept = reg.intercept_
    elif tester == 'linear':
        label = type + ', l'  + ', ' + str(DOFOffset) + ',<' + str(ignoreTargetMotionTimesLessThan) 
        modelCoeff = reg.coef_
        modelIntercept = reg.intercept_
    elif tester == 'ridge':
        label = type + ', r'  + ', ' + str(DOFOffset) + ',<' + str(ignoreTargetMotionTimesLessThan) 
        modelCoeff = reg.coef_
        modelIntercept = reg.intercept_

    returnDict = {
        'True Angles': trueAngles,
        'Est Angles': estAngles,
        'Angular Errors': angularErrors,
        'Times': times,
        'Percent Errors': percentErrors,
        'Average Percentage Error': avgPercentError,
        'Angular Accuracy': angularAccuracyMetric,
        'Label': label,
        'Coeff': modelCoeff,
        'Intercept': modelIntercept,
        'MinDOF': minDof1,
        'MaxDOF': maxDof1,
        'DOFOffset': DOFOffset,
        'PredCursorPos': Y_pred,
        'TestCursorPos': Y_test_true,
        'True Mags': magsTrue,
        'Est Mags': magsEst,
        'Mag Errors': trialMagDifferences,
        'MagPercent Errors': percentErrorsMag,
        'Average Percentage Error': avgPercentErrorMag,
        'Mag Accuracy': magAccuracyMetric,
        'Model': reg,
        'Score': score,
        "X_train": X_train,
        "Y_train": Y_train,
        

    } 
    return returnDict



## -- TRAIN ON EACH LIMB --

def fitModelToEachLimb(compPca,savePath,colorMap = None,plot = False,DOFOffset = 0.03,ignoreTargetMotionTimesLessThan = 600,analysis = False,kFolds = None,participantNo = 0,dummyDecoder = False):
    if analysis:
        path = "/Users/ashwin/Documents/Y4 project Brain Human Interfaces/General 4th year Github repo/BrainMachineInterfaces/Experiment_pointer/DataAnalysis"
        os.chdir(path)
    rigidBodies1, cursorPos1,cursorVel1,goCues1,targetHits1,timeStamps1, minDof1,maxDof1,c,targetMissAppearTimes1 = processTrialData(savePath + "_test",DOFOffset)# make this test 
    rigidBodies2, cursorPos2,cursorVel2,goCues2,targetHits2,timeStamps2, minDof2,maxDof2,d,targetMissAppearTimes2 = processTrialData(savePath + "_training1",DOFOffset)
    rigidBodies3, cursorPos3,cursorVel3,goCues3,targetHits3,timeStamps3, minDof3,maxDof3,e,targetMissAppearTimes3 = processTrialData(savePath + "_training2",DOFOffset)
    rigidBodies4, cursorPos4,cursorVel4,goCues4,targetHits4,timeStamps4, minDof4,maxDof4,f,targetMissAppearTimes4 = processTrialData(savePath + "_training3",DOFOffset)
    rigidBodies5, cursorPos5,cursorVel5,goCues5,targetHits5,timeStamps5, minDof5,maxDof5,g,targetMissAppearTimes5 = processTrialData(savePath + "_training4",DOFOffset)

    # This needs to be updated for k folds
    if kFolds is not None:
        if kFolds == 1:
            rigidBodyVectorTraining = np.concatenate((rigidBodies2,rigidBodies3,rigidBodies4,rigidBodies5), axis = 0)
            cursorPosTraining = np.concatenate((cursorPos2,cursorPos3,cursorPos4,cursorPos5),axis = 0)
            maxDofTraining = np.max(np.concatenate((maxDof2,maxDof3,maxDof4,maxDof5)).reshape(-1,4),1)
            minDofTraining = np.min(np.concatenate((minDof2,minDof3,minDof4,minDof5)).reshape(-1,4),1)

            rigidBodyVectorTest = rigidBodies1
            cursorPosTest = cursorPos1
            goCuesTest = goCues1
            targetHitsTest = targetHits1
            timeStampsTest = timeStamps1

        elif kFolds == 2:
            rigidBodyVectorTraining = np.concatenate((rigidBodies3,rigidBodies4,rigidBodies5,rigidBodies1), axis = 0)
            cursorPosTraining = np.concatenate((cursorPos3,cursorPos4,cursorPos5,cursorPos1),axis = 0)
            maxDofTraining = np.max(np.concatenate((maxDof3,maxDof4,maxDof5,maxDof1)).reshape(-1,4),1)
            minDofTraining = np.min(np.concatenate((minDof3,minDof4,minDof5,minDof1)).reshape(-1,4),1)

            rigidBodyVectorTest = rigidBodies2
            cursorPosTest = cursorPos2
            goCuesTest = goCues2
            targetHitsTest = targetHits2
            timeStampsTest = timeStamps2
        
        elif kFolds == 3:
            rigidBodyVectorTraining = np.concatenate((rigidBodies4,rigidBodies5,rigidBodies1,rigidBodies2), axis = 0)
            cursorPosTraining = np.concatenate((cursorPos4,cursorPos5,cursorPos1,cursorPos2),axis = 0)
            maxDofTraining = np.max(np.concatenate((maxDof4,maxDof5,maxDof1,maxDof2)).reshape(-1,4),1)
            minDofTraining = np.min(np.concatenate((minDof4,minDof5,minDof2,minDof2)).reshape(-1,4),1)

            rigidBodyVectorTest = rigidBodies3
            cursorPosTest = cursorPos3
            goCuesTest = goCues3
            targetHitsTest = targetHits3
            timeStampsTest = timeStamps3
        
        elif kFolds == 4:
            rigidBodyVectorTraining = np.concatenate((rigidBodies5,rigidBodies1,rigidBodies2,rigidBodies3), axis = 0)
            cursorPosTraining = np.concatenate((cursorPos5,cursorPos1,cursorPos2,cursorPos3),axis = 0)
            maxDofTraining = np.max(np.concatenate((maxDof5,maxDof1,maxDof2,maxDof3)).reshape(-1,4),1)
            minDofTraining = np.min(np.concatenate((minDof5,minDof1,minDof2,minDof3)).reshape(-1,4),1)

            rigidBodyVectorTest = rigidBodies4
            cursorPosTest = cursorPos4
            goCuesTest = goCues4
            targetHitsTest = targetHits4
            timeStampsTest = timeStamps4

        elif kFolds == 5:
            rigidBodyVectorTraining = np.concatenate((rigidBodies1,rigidBodies2,rigidBodies3,rigidBodies4), axis = 0)
            cursorPosTraining = np.concatenate((cursorPos1,cursorPos2,cursorPos3,cursorPos4),axis = 0)
            maxDofTraining = np.max(np.concatenate((maxDof1,maxDof2,maxDof3,maxDof4)).reshape(-1,4),1)
            minDofTraining = np.min(np.concatenate((minDof1,minDof2,minDof3,minDof4)).reshape(-1,4),1)

            rigidBodyVectorTest = rigidBodies5
            cursorPosTest = cursorPos5
            goCuesTest = goCues5
            targetHitsTest = targetHits5
            timeStampsTest = timeStamps5

    else:

        rigidBodyVectorTraining = np.concatenate((rigidBodies2,rigidBodies3,rigidBodies4,rigidBodies5), axis = 0)
        cursorPosTraining = np.concatenate((cursorPos2,cursorPos3,cursorPos4,cursorPos5),axis = 0)
        maxDofTraining = np.max(np.concatenate((maxDof2,maxDof3,maxDof4,maxDof5)).reshape(-1,4),1)
        minDofTraining = np.min(np.concatenate((minDof2,minDof3,minDof4,minDof5)).reshape(-1,4),1)

        rigidBodyVectorTest = rigidBodies1
        cursorPosTest = cursorPos1
        goCuesTest = goCues1
        targetHitsTest = targetHits1
        timeStampsTest = timeStamps1
    
    # Iterate over all limbs
    if dummyDecoder:
        r2AdjVals = np.zeros(20)
    else:
        r2AdjVals = np.zeros(19)

    for limb in range(19):
        # # # Angles only: only get the left hand

        # Extract limb for test and training
        if limb == 20:

            X_train = np.random.shuffle(rigidBodyVectorTraining[:,3*6+3:3*6+6])
            X_test_linear = np.random.shuffle(rigidBodyVectorTest[:,5*6+3:5*6+6])
        else:
            X_train = rigidBodyVectorTraining[:,limb*6+3:limb*6+6]
            X_test_linear = rigidBodyVectorTest[:,limb*6+3:limb*6+6]
    
        # Fit model of training data to limb
        Y_train = cursorPosTraining
        reg  = linear_model.LinearRegression().fit(X_train, Y_train)
        

        # predict for test data
        Y_test_linear = reg.predict(X_test_linear)
        score = reg.score(X_test_linear,cursorPosTest)

        Y_pred = Y_test_linear
        Y_test_true = cursorPosTest


        scoreLabel = 'l' +  ', ' + str(DOFOffset)

        # now calculate average angular error on the test dataset
        if goCuesTest[0] == targetHitsTest[0]:
            del goCuesTest[0]
            del targetHitsTest[0]

        trueAngles,estAngles, angularErrors,times,magsTrue,magsEst,trialMagDifferences = feedTargetMotionCursorPos(Y_test_true,Y_pred,goCuesTest,targetHitsTest,timeStampsTest,ignoreTargetMotionTimesLessThan)
        percentErrors = 100 * [abs((abs(estAngles[i] - trueAngles[i]))/trueAngles[i]) for i in range(0,len(trueAngles))]
        avgPercentError = np.average((percentErrors)) 
        percentErrorsMag = 100 * [abs(abs((magsEst[i] - magsTrue[i]))/magsTrue[i]) for i in range(0,len(magsTrue))]
        avgPercentErrorMag = np.average((percentErrorsMag)) 
        #print('error', avgPercentError)
        angularAccuracyMetric = 1 - np.abs(avgPercentError) / 1
        magAccuracyMetric = 1 - np.abs(avgPercentErrorMag) /1

        #label = type + ', l_PCA'  + str(compPca) + ', ' + str(DOFOffset) + ',<' + str(ignoreTargetMotionTimesLessThan) 
        modelCoeff = reg.coef_
        modelIntercept = reg.intercept_

        #label = type + ', l'  + ', ' + str(DOFOffset) + ',<' + str(ignoreTargetMotionTimesLessThan) 
        modelCoeff = reg.coef_
        modelIntercept = reg.intercept_

    
        modelDict = {
            'True Angles': trueAngles,
            'Est Angles': estAngles,
            'Angular Errors': angularErrors,
            'Times': times,
            'Percent Errors': percentErrors,
            'Average Percentage Error': avgPercentError,
            'Angular Accuracy': angularAccuracyMetric,
            'Coeff': modelCoeff,
            'Intercept': modelIntercept,
            'MinDOF': minDof1,
            'MaxDOF': maxDof1,
            'DOFOffset': DOFOffset,
            'PredCursorPos': Y_pred,
            'TestCursorPos': Y_test_true,
            'True Mags': magsTrue,
            'Est Mags': magsEst,
            'Mag Errors': trialMagDifferences,
            'MagPercent Errors': percentErrorsMag,
            'Average Percentage Error': avgPercentErrorMag,
            'Mag Accuracy': magAccuracyMetric,
            'Model': reg,
            'Score': score,
            "X_train": X_train,
            "Y_train": Y_train,
            

        } 

        # Calc adj r2 value and return this for each limb
        r2AdjVals[limb] = plotTrajectories(modelDict,kFolds-1)


    return r2AdjVals



# BELOW FUNCTIONS FOR DATA ANALYSIS

def readIndividualTargetMovements(processedDataDict):
    returnDict = {
        'rigidBodyData': [], 
        'cursorPosData': [], 
        'cursorVelData': [],
        'timestamps': []
    }
    for i in range(0,len(processedDataDict['goCues'])-1):
        startTime = processedDataDict['timestamps'][processedDataDict['goCues'][i]]
        returnDict['rigidBodyData'].append(processedDataDict['rigidBodyData'][processedDataDict['goCues'][i]:processedDataDict['targetReached'][i],:].transpose())
        returnDict['cursorPosData'].append(processedDataDict['cursorPos'][processedDataDict['goCues'][i]:processedDataDict['targetReached'][i],:].transpose())
        returnDict['cursorVelData'].append(processedDataDict['cursorVel'][processedDataDict['goCues'][i]:processedDataDict['targetReached'][i],:].transpose())
        returnDict['timestamps'].append(processedDataDict['timestamps'][processedDataDict['goCues'][i]:processedDataDict['targetReached'][i]] - startTime)    

    return returnDict

# Fetch decoder data
from sklearn.metrics import r2_score
from DecoderFunctions.decoderTrainingFunctions import processTrialData, colorMap
from matplotlib.gridspec import GridSpec

def plotTrajectories(decoder,split,decoderType = None,plottingOn = False):

    # Get arbitrary training files
    participantIdentifiers = dataExtractor.extractSpecificParticipantFiles(trialType = "_training1")
    noParticipants = len(participantIdentifiers)
    noDecoders = 7
    K = 5 # number of folds in K folds
    # Remove training part off each file
    for i in range(len(participantIdentifiers)):
        participantIdentifiers[i] = participantIdentifiers[i].replace("_training1", "")

    cmap_ = [cmap[3],cmap[0],cmap[1]]
    """This function takes in a decoder dict object which is outputted from the fitModelToData function
    It then generates plots of the actual test trajectories (ground truth) against test trajectories predicted
    from the model.

    It also returns an adjusted r2 score, where in each trajectory, the difference between the predicted trajectory
    and actual trajectory start point has been subtracted from the predicted trajectory. This R2 score
    better reflects the model performance decoding a user's movement from a known start position.

    Inputs:
        @param: decoder - A dict object specifying properties of the decoder gathered 
        from the fitModelToData function
        @param k - A int specifying how many trajectories to plot, must be less or equal to the total number of trajectories
        @param split: the trial that is the test [0,1,2,3,4]
        @param decoderType - Char/String specifying decoder type for saved, if none then plots are not saved

    """
    # Extract true and predicted cursor positions for all test data
    Y_pred = np.zeros(decoder['PredCursorPos'].shape)
    correctY = np.zeros(decoder['TestCursorPos'].shape)
    Y_pred_ = decoder['PredCursorPos']
    correctY_ = decoder['TestCursorPos']

    # In pygame y axis is inverted so this is accounted for by mapping normalised y pos from 0 - 1 to 1 - 0
    correctY[:,1] = 1  - correctY_[:,1] 
    Y_pred[:,1] = 1 - Y_pred_[:,1] 
    correctY[:,0] = correctY_[:,0]
    Y_pred[:,0] = Y_pred_[:,0]

    actualPos = []
    predPos = []
    
    # Retrieve test go cue data
    mapSplitToTerm = { 0: "_test", 1: "_training1", 2: "_training2", 3: "_training3", 4: "_training4"

    }
    
    identifierTrial = mapSplitToTerm[split]
    rigidBodies1, cursorPos1,cursorVel1,goCues1,targetHits1,timeStamps1, minDof1,maxDof1,c,failedTargetAppearTimes = processTrialData(participantIdentifiers[0] + identifierTrial,0.01)# make this test 
    if plottingOn:
        fig = plt.figure(figsize=(12,4))
        gs = GridSpec(1, 3, figure=fig, width_ratios=[0.49,0.02, 0.49], height_ratios=[1])
        ax0 = fig.add_subplot(gs[0, 0])
        
    #print("Plotting trajectories without compensating for offset")

    # Gather each trajectory
    for i in range(0,len(goCues1)-1):
        plotFrom = goCues1[i]
        plotUntil = targetHits1[i]

        # Upload trajectory path to list for future stage
        actualPos.append(correctY[plotFrom:plotUntil])
        predPos.append(Y_pred[plotFrom:plotUntil])

        # plot first k trajectories
        if plottingOn:
            if i in [0,7,8]:
                if i < k:
                    ax0.plot(correctY[plotFrom:plotUntil,0],correctY[plotFrom:plotUntil,1],color = cmap_[i%len(cmap_)])
                    ax0.plot(Y_pred[plotFrom:plotUntil,0],Y_pred[plotFrom:plotUntil,1], "--",color = cmap_[i%len(cmap_)])
                    
                    # Plot start and end markers (for i = 0 this in addition gives a label in the legend )
                    if i == 0:
                        ax0.plot([],[],color = 'k',label = "Ground Truth Trajectory")
                        ax0.plot([],[], "--", color = 'k',label="Predicted Trajectory")
                        ax0.scatter(correctY[plotFrom,0], correctY[plotFrom,1],s=250, marker=".", color = 'g',label = 'Actual cursor start position')
                        # plt.scatter(correctY[plotUntil,0], correctY[plotUntil,1], s=100, marker="D", color = 'g')
                        ax0.scatter(Y_pred[plotFrom,0], Y_pred[plotFrom,1],s=250, marker=".", color = 'r',label = 'Estimated cursor start position')
                        # plt.scatter(Y_pred[plotUntil,0], Y_pred[plotUntil,1], s=60, marker="D", color = 'r')
                    else:
                        ax0.scatter(correctY[plotFrom,0], correctY[plotFrom,1],s=250, marker=".", color = 'g')
                        # plt.scatter(correctY[plotUntil,0], correctY[plotUntil,1], s=100, marker="D", color = 'g')
                        ax0.scatter(Y_pred[plotFrom,0], Y_pred[plotFrom,1],s=250, marker=".", color = 'r')
                        # plt.scatter(Y_pred[plotUntil,0], Y_pred[plotUntil,1], s=60, marker="D", color = 'r')
    if plottingOn:
        # Remove top and right spines for the first plot
        ax0.spines['top'].set_visible(False)
        ax0.spines['right'].set_visible(False)

        # Set plotting configuration
        ax0.set_xlabel(r'$X_{pos}$',fontsize = 25,fontweight='bold')
        ax0.set_ylabel(r'$Y_{pos}$', fontsize = 25,fontweight='bold')
        ax0.set_xticks([0,1],fontsize = 15)
        ax0.set_yticks([0,1],fontsize = 15)
        ax0.text(-0.3, 1.3, 'A)', ha='center', va='top', fontsize=30, fontweight = "bold")
        #plt.title('Trajectories showing actual and estimated cursor position for each target aquisition performed in test set. \n Each trajectory is shown in a different colour and position estimates are derived from set (b) of rigid bodies',fontsize = 15)
        
        pf.defaultPlottingConfiguration(ax0)
        ax0.legend(loc="lower right",fontsize=50,prop={'weight': 'bold', 'size': 15})
        # Save the file if necessary
        if decoderType is not None:
            #plt.savefig(saveGameLocation + "_Decoder {} trajectories_noOffset".format(decoderType))
            ax1 = fig.add_subplot(gs[0, 2])

        #print("Plotting trajectories  compensating for offset")
    for i in range(0,len(goCues1)-1): # len(goCues1) len(colorMap)
        plotFrom = goCues1[i]
        plotUntil = targetHits1[i]
        # if np.sqrt(np.square(correctY[plotUntil,0] - correctY[plotFrom,0]) + np.square(correctY[plotUntil,1] - correctY[plotFrom,1])) < 0.15:
        #     break

        offset = Y_pred[plotFrom] - correctY[plotFrom]
        Y_pred[plotFrom:plotUntil] = Y_pred[plotFrom:plotUntil] - offset

        if plottingOn:
            if i in [0,7,8]:
                if i == 0:
                    ax1.plot([],[],color = 'k',label = "Ground Truth Trajectory")
                    ax1.plot([],[], "--", color = 'k',label="Predicted Trajectory")
                    ax1.scatter([], [],s=250, marker=".", color = 'b',label = 'Cursor Start Position')
                    ax1.plot(correctY[plotFrom:plotUntil,0],correctY[plotFrom:plotUntil,1],color = cmap_[i%len(cmap_)])
                    ax1.plot(Y_pred[plotFrom:plotUntil,0],Y_pred[plotFrom:plotUntil,1], "--", color = cmap_[i%len(cmap_)])
                    ax1.scatter(correctY[plotFrom,0], correctY[plotFrom,1],s=250, marker=".", color = 'b')
                elif i < k+1:
                    ax1.plot(correctY[plotFrom:plotUntil,0],correctY[plotFrom:plotUntil,1],color = cmap_[i%len(cmap_)])
                    ax1.plot(Y_pred[plotFrom:plotUntil,0],Y_pred[plotFrom:plotUntil,1], "--",color = cmap_[i%len(cmap_)])
                    #if i == 0:
                        #plt.scatter(correctY[plotFrom,0], correctY[plotFrom,1],s=250, marker=".", color = 'b',label = 'Cursor start position')
                        # plt.scatter(correctY[plotUntil,0], correctY[plotUntil,1], s=100, marker="D", color = 'g')
                        #plt.scatter(Y_pred[plotFrom,0], Y_pred[plotFrom,1],s=250, marker=".", color = 'r',label = 'Estimated cursor start position')
                        # plt.scatter(Y_pred[plotUntil,0], Y_pred[plotUntil,1], s=60, marker="D", color = 'r')
                    #else:
                    ax1.scatter(correctY[plotFrom,0], correctY[plotFrom,1],s=250, marker=".", color = 'b')
                    # plt.scatter(correctY[plotUntil,0], correctY[plotUntil,1], s=100, marker="D", color = 'g')
                    #plt.scatter(Y_pred[plotFrom,0], Y_pred[plotFrom,1],s=250, marker=".", color = 'r')
                    # plt.scatter(Y_pred[plotUntil,0], Y_pred[plotUntil,1], s=60, marker="D", color = 'r')

    if plottingOn:
        ax1.set_xlabel(r'$X_{pos}$',fontsize = 20)
        ax1.set_ylabel(r'$Y_{pos}$',fontsize = 20)
        #plt.title('Trajectories showing actual and estimated cursor position for each target aquisition performed in test set. \n Each trajectory is shown in a different colour and position estimates are derived from set (b) of rigid bodies',fontsize = 15)
        
        ax1.set_xticks([0,1],fontsize = 15)
        ax1.set_yticks([0,1],fontsize = 15)

        pf.defaultPlottingConfiguration(ax1)
        ax1.legend(loc="lower right",fontsize=50,prop={'weight': 'bold', 'size': 15})
        ax1.text(-0.1, 1.3, 'B)', ha='center', va='top', fontsize=30, fontweight = "bold")
        # Remove top and right spines for the first plot
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # Save the file if necessary
        if decoderType is not None:
            #plt.savefig("DecoderF-example-offsetsuitability.pdf",bbox_inches='tight')
            #plt.savefig(saveGameLocation + "_Decoder {} trajectories_withOffset.svg".format(decoderType))
            plt.show()

    # then subtract mean
    allPredPosX = []
    allPredPosY = []
    allTruePosX = []
    allTruePosY = []
    r2AdjTrajVals = []
    r2TrajVals_ = []
    for idx in range(len(actualPos)-1):
        truePosTrajectory = actualPos[idx]
        predPosTrajectory = predPos[idx]
        # plt.plot(truePosTrajectory[:,0],truePosTrajectory[:,1])
        # plt.plot(predPosTrajectory[:,0],predPosTrajectory[:,1])
        # plt.show()
        dist = calcDist(truePosTrajectory[0,:],truePosTrajectory[-1,:])
        #print(dist)
        if dist > 500:
            r2TrajectoryScore = float(r2_score(truePosTrajectory, predPosTrajectory))
            #print(r2TrajectoryScore)
            r2TrajVals_.append(r2TrajectoryScore)
        offset = predPosTrajectory[0] - truePosTrajectory[0]
        predPosTrajectory = predPosTrajectory - offset
        if dist > 500:
            r2TrajectoryScoreAdj = float(r2_score(truePosTrajectory, predPosTrajectory))
            r2AdjTrajVals.append(r2TrajectoryScoreAdj)
        for val in truePosTrajectory:
            allTruePosX.append(val[0])
            allTruePosY.append(val[1])
        for val in predPosTrajectory:
            allPredPosX.append(val[0])
            allPredPosY.append(val[1])

    pos_true = np.zeros((len(allTruePosX),2))
    pos_pred = np.zeros(pos_true.shape)
    pos_true[:,0] = allTruePosX
    pos_true[:,1] = allTruePosY
    pos_pred[:,0] = allPredPosX
    pos_pred[:,1] = allPredPosY
    r2Score = r2_score(pos_true, pos_pred) # this is adj r2
    #print("R2 val for this decoder: {}".format(r2Score))

    #return pos_true, pos_pred, r2Score, r2TrajVals_, r2AdjTrajVals
    return r2Score


def calcDist(startPos,endPos):
    ranges = [1100+800,800+225]
    return np.sqrt(np.sum([ (ranges[i] * (endPos[i] - startPos[i])) ** 2 for i in range(len(startPos))]))





def analyseNotebook(notebookInputPath, notebookOutputPath):

    

    # Load the notebook
    print("Opening Notebook ...")
    with open(notebookInputPath) as f:
        nb = nbformat.read(f, as_version=4)

    # Set up the ExecutePreprocessor
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

    # Execute the notebook
    print("Executing Notebook ...")
    ep.preprocess(nb, {'metadata': {'path': './'}})  # Specify the path for any relative paths in the notebook

    # Save the executed notebook
    exporter = NotebookExporter()
    body, _ = exporter.from_notebook_node(nb)

    print("Exporting Notebook ...")
    with open(notebookOutputPath, 'w') as f:
        f.write(body)