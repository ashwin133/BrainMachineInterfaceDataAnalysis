"""
Statistical tests to test hypothesis for left hand decoder
"""

# Run paired t-test to evaluate if left hand shows more variability throughout trial


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



# Fetch key decoding data for variability analysis rigid body data is shape: T x noDOF x noDecoders x noParticipants
rigidBodyDecoderData, scores, noParticipants = dataExtractor.retrieveDecoderData()
labels = ['PVS', 'AB', 'CH', 'NCK', 'HD', 'LSD', 'LUA', 
                      'LFA', 'LHD', 'RSD', 'RUA', 'RFA', 'RHD',  'LTH', 'LSN', 
                      'LFT', 'RTH', 'RSN', 'RFT']

# Calculate variability values of decoder data
rmsValues = variabilityAnalysis.calculateVariabilityScoresInSegmentedTrial(rigidBodyDecoderData,includePositions=True)
print(rmsValues.shape)
#  should be of shape noSegments x noDOF x noDecoders x noParticipants

# Calculate variability values of individual body parts by summing over dof for each body part
rmsValues = np.sum(rmsValues.reshape(5,19,6,7,noParticipants),axis = 2).reshape(5,19,7,noParticipants)

# Find rmsValues for decoder G for the left hand

leftHandIdx = labels.index('LHD')
decoderGIdx = 2

rmsValuesLeftHand = rmsValues[:,leftHandIdx,decoderGIdx,:]

# Resize rms values to correct range
maxVal = np.max(rmsValuesLeftHand)
rmsValuesLeftHand = rmsValuesLeftHand / maxVal

# Average first two segments variability
startRMSLeftHand = np.average(rmsValuesLeftHand[0:2,:],axis = 0)

# Average last two segments
endRMSLeftHand = np.average(rmsValuesLeftHand[3:5,:],axis = 0)


avgDiff = np.average(endRMSLeftHand - startRMSLeftHand)

percentAvg = 100 * avgDiff / np.average(startRMSLeftHand)

# Test if both start and average variabilities follow normal distribution, otherwise cannot run paired t test

# Run the Shapiro-Wilk test to test normality
shapiroResultStartVariability = stats.shapiro(startRMSLeftHand)
shapiroResultEndVariability = stats.shapiro(endRMSLeftHand)

# Raise an exception if the Shapiro-Wilk test fails
if shapiroResultStartVariability.pvalue < 0.05 or shapiroResultEndVariability.pvalue < 0.05:
    raise("You should not run a paired t-test here as one variability distribution violates normality assumption, bypass this error if needed")
else:
    print("Both distributions pass the Shapiro test, the p values are {} and {}".format(shapiroResultStartVariability.pvalue,shapiroResultEndVariability.pvalue))

variabilityPairedTTestResults = stats.ttest_rel(startRMSLeftHand,endRMSLeftHand)

pValue = variabilityPairedTTestResults.pvalue

print("Paired-ttest p value: {}".format(pValue))

if pValue < 0.05:
    print("Result signficant at the 5 % level")
else:
    print("Result not significant at the 5% level")

# Plot a box plot

# Define colours
red = (245/255,5/255,5/255) # (RGB) or F50505 (Hex)
colors = [red,red]

# Define other parameters for function
xTickList = ['Segments 1 and 2', 'Segments 4 and 5']
variabilities = [startRMSLeftHand,endRMSLeftHand]

# Define figure properties
fig = plt.figure(figsize=(10,6))
ax = plt.gca()

# Create box plot
plottingFuncs.createBoxPlot(ax, variabilities, colors, xTickList, 'Segments', 'Variability metric')

# Create message for p value
msgPValue = "P value for increase of left hand variability for decoder G: {}. \n Percentage increase between start and end segments: +{}%".format(round(pValue,3),round(percentAvg,1))
# Make title
plt.title("Box plot of variability of left hand in starting and ending segments \n" + msgPValue, fontsize = 20)

# Plot legend 
plt.legend(fontsize = 15)

# Ensure everything is visible
fig.tight_layout() 

plt.show()



