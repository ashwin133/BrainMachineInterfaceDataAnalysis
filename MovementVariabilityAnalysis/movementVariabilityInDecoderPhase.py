"""
Look at superstitous movements in decoder trials
"""


# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import sys
from scipy import stats

# add current path to system PATH 
sys.path.insert(0,'/Users/ashwin/Documents/Y4 project Brain Human Interfaces/General 4th year Github repo/BrainMachineInterfaceDataAnalysis')

# Import user defined libraries
import DataExtraction.extractRawData as dataExtractor
from BasicAnalytics import targetAcqusitionPlotting as targetPlotter
from BasicAnalytics import variabilityAnalysis 
from BasicAnalytics import plottingFuncs



# Fetch key training data for variability analysis rigid body data is shape: T x noDOF x noDecoders x noParticipants
rigidBodyDecoderData, scores, noParticipants = dataExtractor.retrieveDecoderData()
labels = ['PVS', 'AB', 'CH', 'NCK', 'HD', 'LSD', 'LUA', 
                      'LFA', 'LHD', 'RSD', 'RUA', 'RFA', 'RHD',  'LTH', 'LSN', 
                      'LFT', 'RTH', 'RSN', 'RFT']


rotationIndicies = []

# Add all suitable rotation indices so that array is [3,4,5,9,10,11....114]
# Start at 3, increment by 6 for each pattern
for i in range(3, 114, 6):  
    # Add three consecutive numbers 
    rotationIndicies.extend([i, i + 1, i + 2])

# Calculate variability values of individual body parts by summing over dof for each body part
timeSeriesRigidBodyData = np.sum(rigidBodyDecoderData[:,rotationIndicies,:,:].reshape(-1,19,3,7,noParticipants),axis = 2).reshape(-1,19,7,noParticipants)

# average over participants
timeSeriesRigidBodyData = np.average(timeSeriesRigidBodyData,axis = 3) # now of shape T x DOF x Decoders
maxVal = np.max(timeSeriesRigidBodyData)
timeSeriesRigidBodyData = timeSeriesRigidBodyData / maxVal
N = 19

# # For G
# timeSeriesDiffRigidBodyData = variabilityAnalysis.centralDifference(timeSeriesRigidBodyData[:,:,2,:])
# plt.plot(np.abs(timeSeriesDiffRigidBodyData[:,[8,12],1]), label = ['Left', 'Right'])
# plt.legend()
# plt.show()

# Create N vertical subplots
fig, axs = plt.subplots(N, 1, figsize=(10, 8))  # Adjust figure size as needed

for i in range(N):
    # Plot each time series in its subplot
    axs[i].plot(timeSeriesRigidBodyData[:,i,2])
    axs[i].set_ylabel(labels[i])
    plt.xlim(500,5500)

# Set common labels
for ax in axs:
    ax.set_xlabel('Time')
plt.tight_layout()  # Adjust subplots to fit in the figure window
plt.show()

# Calculate variability values of decoder data
rmsValues = variabilityAnalysis.calculateVariabilityScoresInSegmentedTrial(rigidBodyDecoderData)
print(rmsValues.shape)
#  should be of shape noSegments x noDOF x noDecoders x noParticipants

# Calculate variability values of individual body parts by summing over dof for each body part
rmsValues = np.sum(rmsValues.reshape(5,19,3,7,noParticipants),axis = 2).reshape(5,19,7,noParticipants)

# Resize rms values to correct range
maxVal = np.max(rmsValues)
rmsValues = rmsValues / maxVal

# Average over participants
rmsValuesAveragedOverParticipants = np.average(rmsValues,axis = 3)

# List of each decoders
decoderList = ['E','F','G','H','I','J','K']
rangeControl = [1,1,1,0,0,0,1,1,0,0,0]
usedRangeControl = [0,1,2,6,7]
noRangeControl = [3,4,5,8,9,10]
label = ["Segment 1", "Segment 2", "Segment 3", "Segment 4", "Segment 5"]
bodyPartlabels = ['PVS', 'AB', 'CH', 'NCK', 'HD', 'LSD', 'LUA', 
'LFA', 'LHD', 'RSD', 'RUA', 'RFA', 'RHD',  'LTH', 'LSN', 
'LFT', 'RTH', 'RSN', 'RFT']


# Create a list of numbers from 0 to 1 to represent points in the colormap
color_range = np.linspace(0, 1, 5)

# Use a built-in colormap (e.g., 'coolwarm' which transitions from blue to red)
colormap = plt.cm.coolwarm

# Generate the list of colors by mapping the color range through the colormap
colors = [colormap(x) for x in color_range]

# Now, colors is a list of RGBA colors transitioning from blue to red
print(colors)


# Plot for each decoder trial
for idx,decoder in enumerate(decoderList):

    fig = plt.figure(figsize=(12,5))
    plt.plot(rmsValuesAveragedOverParticipants[0,:,idx].T,label = label[0],color = colors[0], alpha = 1)
    plt.plot(rmsValuesAveragedOverParticipants[1,:,idx].T,label = label[1],color = colors[1], alpha = 0.9)
    plt.plot(rmsValuesAveragedOverParticipants[2,:,idx].T,label = label[2],color = colors[2], alpha = 0.8)
    plt.plot(rmsValuesAveragedOverParticipants[3,:,idx].T,label = label[3],color = colors[3], alpha = 0.7)
    plt.plot(rmsValuesAveragedOverParticipants[4,:,idx].T,label = label[4],color = colors[4], alpha = 0.6)
    
    x = range(0, 19)


    # Set custom x-tick labels
    plt.xticks(x, labels)
    plt.xlabel("Body Parts", fontsize = 20)
    plt.ylabel("Average variability", fontsize = 20)
    plt.title("Average variability metrics for all body parts by segment for decoder {}".format(decoder), fontsize = 20)
    plt.legend(fontsize = 15)
    plt.tick_params(axis='y', which='major', labelsize=14)
    plt.tick_params(axis='x', which='major', labelsize=14)
    plt.legend()
    plt.show()




from scipy import stats



scoresRangeControl = scores[usedRangeControl,:]
scoresNoRangeControl = scores[noRangeControl,:]

avgScoreRangeControl = np.average(scoresRangeControl,axis = 0)
avgScoreNoRangeControl = np.average(scoresNoRangeControl, axis = 0)



multiplier = avgScoreRangeControl/ avgScoreNoRangeControl
print("multiplier:",multiplier)



for i in range(7):
    fig = plt.figure(figsize = (8,6))
    ax = plt.gca()
    scoresForBoxPlot = [scoresRangeControl[:,i],scoresNoRangeControl[:,i]]
    xTickList = ['range', 'no range']
    pVal = stats.ttest_ind(scoresRangeControl[:,i],scoresNoRangeControl[:,i]).pvalue
    colors = ['red', 'red']
    plottingFuncs.createBoxPlot(ax, scoresForBoxPlot, colors, xTickList, 'Decoder', 'Score')
    plt.title("Box Plot of scores for decoder {} with \n and without range control  p value: {}".format(decoderList[i],round(pVal,3)), fontsize = 20)
    # Middle of box plot is median and line is average so there may be differences
    plt.legend(fontsize = 15)
    fig.tight_layout() 
plt.show()
