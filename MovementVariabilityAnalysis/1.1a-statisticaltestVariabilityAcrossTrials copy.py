"""
Statistical tests (paired t-test) to show that variability decreases over trials
"""
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import sys
from scipy import stats
from tol_colors import tol_cmap, tol_cset
cmap = tol_cset('bright')
import numpy as np
import matplotlib.pyplot as plt


# # Using seaborn's style
# plt.style.use('seaborn')
# width = 345

tex_fonts = {

    "text.usetex": True,
    "font.family": "serif",

    "axes.labelsize": 12,
    "font.size": 12,

    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
}

plt.rcParams.update(tex_fonts)

# add current path to system PATH 
sys.path.insert(0,'/Users/ashwin/Documents/Y4 project Brain Human Interfaces/General 4th year Github repo/BrainMachineInterfaceDataAnalysis')

# Import user defined libraries
import DataExtraction.extractRawData as dataExtractor
from BasicAnalytics import targetAcqusitionPlotting as targetPlotter
from BasicAnalytics import variabilityAnalysis 
from BasicAnalytics import plottingFuncs


# Fetch key training data for variability analysis
rigidBodyTrain, scores, noParticipants = dataExtractor.retrieveTrainingData()

# Calculate variability values of training data
rmsValues = variabilityAnalysis.calculateVariabilityScores(rigidBodyTrain,includePositions=False)
print(rmsValues.shape)

# Calculate variability values of individual body parts by summing over dof for each body part
rmsValues = np.sum(rmsValues.reshape(19,3,noParticipants*5),axis = 1).reshape(19,noParticipants*5)

# Normalisation and sum variability values across DOF
summedRMSvaluesAcrossDOF = np.sum(rmsValues.reshape(19,5,noParticipants),axis = 0)
summedRMSvaluesAcrossDOF = summedRMSvaluesAcrossDOF / np.max(summedRMSvaluesAcrossDOF)

plt.plot(summedRMSvaluesAcrossDOF,label = [str(i) for i in range(11) ])
plt.legend()
plottingFuncs.defaultPlottingConfiguration(plt.gca(),maxYTicks=4,ylabel = "Variability",xlabel = "Trial")
plt.show()

# # Run paired t test on average variability of trial 1 and 2 against average variability of trial 4 and 5
# # Implementation from https://pythonfordatascienceorg.wordpress.com/paired-samples-t-test-python/
# for j in range(summedRMSvaluesAcrossDOF.shape[1]):
#     plt.scatter([0,1,2,3,4],summedRMSvaluesAcrossDOF[:,j])
# plt.show()
print(stats.pearsonr([0]*11 + [1]*11 + [2]*11 + [3] * 11 + [4] * 11,summedRMSvaluesAcrossDOF[:,:].reshape(-1)))
statMain,pvalueMain= stats.pearsonr([0]*11 + [1]*11 + [2]*11 + [3] * 11 + [4] * 11,summedRMSvaluesAcrossDOF[:,:].reshape(-1))
# Start average variabilities
averageVariabilitiesStart = np.average(summedRMSvaluesAcrossDOF[0:2,:], axis = 0)

# End average variabilities
averageVariabilitiesEnd = np.average(summedRMSvaluesAcrossDOF[3:5,:], axis = 0)

avgDiff = np.average(averageVariabilitiesEnd - averageVariabilitiesStart)

percentAvg = 100 * avgDiff / np.average(averageVariabilitiesStart)

# Test if both start and average variabilities follow normal distribution, otherwise cannot run paired t test

# Run the Shapiro-Wilk test to test normality
shapiroResultStartVariability = stats.shapiro(averageVariabilitiesStart)
shapiroResultEndVariability = stats.shapiro(averageVariabilitiesEnd)

# Raise an exception if the Shapiro-Wilk test fails
if shapiroResultStartVariability.pvalue < 0.05 or shapiroResultEndVariability.pvalue < 0.05:
    print("Both distributions do not pass the Shapiro test, the p values are {} and {}".format(shapiroResultStartVariability.pvalue,shapiroResultEndVariability.pvalue))
    print("Running a paired t-test is not recommended!!")
    #raise("You should not run a paired t-test here as one variability distribution violates normality assumption, bypass this error if needed")
else:
    print("Both distributions pass the Shapiro test, the p values are {} and {}".format(shapiroResultStartVariability.pvalue,shapiroResultEndVariability.pvalue))

variabilityPairedTTestResults = stats.ttest_rel(averageVariabilitiesStart,averageVariabilitiesEnd)

pValue = variabilityPairedTTestResults.pvalue

print("Paired-ttest p value: {}".format(pValue))

if pValue < 0.05:
    print("Result signficant at the 5 % level")
else:
    print("Result not significant at the 5% level")

# Plot a box plot

# Define colours
red = (245/255,5/255,5/255) # (RGB) or F50505 (Hex)
colors = [cmap[0]]* 5

# Define other parameters for function
xTickList = ['1','2', '3', '4', '5']
variabilities = [summedRMSvaluesAcrossDOF[0,:],summedRMSvaluesAcrossDOF[1,:],summedRMSvaluesAcrossDOF[2,:],summedRMSvaluesAcrossDOF[3,:],summedRMSvaluesAcrossDOF[4,:]]

# Define figure properties
fig = plt.figure(figsize=(10,6))
ax = plt.gca()

# Create box plot
plottingFuncs.createBoxPlot(ax, variabilities, colors, xTickList, 'Trial', 'Variability metric')

# Create message for p value
msgPValue = "P value: {}.".format(round(pvalueMain,3))
# Make title
plt.title("Box and whisker plot of variability of rigid bodies across baseline phase \n" + msgPValue, fontsize = 22)

# Plot legend 
plt.legend(fontsize = 15)

# Ensure everything is visible
fig.tight_layout() 
plt.savefig("StatisticalTest.pdf")
plt.show()

