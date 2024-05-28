"""
Functionality for basic plotting 
"""
from matplotlib.ticker import MaxNLocator
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def createBoxPlot(ax,listOfVars,colorList,xTickList = None,xlabel = None,ylabel = None):
    box = ax.boxplot(listOfVars, patch_artist=True)

    for patch, color in zip(box['boxes'], colorList):
        patch.set_facecolor(color)

    # Customize the whiskers, caps, and median
    for whisker in box['whiskers']:
        whisker.set(color='black', linewidth=1.5)
    for cap in box['caps']:
        cap.set(color='black', linewidth=2)
    for median in box['medians']:
        median.set(color='black', linewidth=2)

    # Adding titles and labels
    if xlabel:
        ax.set_xlabel(xlabel,fontsize = 22,fontweight='bold')
    
    if ylabel:
        ax.set_ylabel(ylabel,fontsize = 22,fontweight='bold')

    ax.tick_params(labelsize = 20)

    if xTickList:
        ax.set_xticklabels(xTickList)

    # Remove top and right spines for the first plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def defaultPlottingConfiguration(ax, labelSize = 24, tickSize = 20,maxXTicks = None, maxYTicks = None,removeSpine = True,removeXTick = False,xlabel = None, ylabel = None,legendSize = 15):

    """Quickly applies requested configuration to figures"""

    ax.xaxis.label.set_fontsize(labelSize)  
    ax.yaxis.label.set_fontsize(labelSize)
    ax.tick_params(axis='both', which='major', labelsize=tickSize)  # 'both' can be replaced with 'x' or 'y' to specify an axis
    
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    # Set maximum number of ticks on y axis
    if maxXTicks is not None:
        ax.yaxis.set_major_locator(MaxNLocator(maxXTicks))  # 
    if maxYTicks is not None:
        ax.yaxis.set_major_locator(MaxNLocator(maxYTicks))  # 

    # Remove top and right spines if requested
    if removeSpine:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
    
    # Remove x ticks and keep the labels only if requested
    if removeXTick:
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='x', which='both', length=0)  # Sets tick mark length to 0 for both axes

    # Sort out legend
    if ax.get_legend() is not None:
        ax.legend(loc='upper right', fontsize = legendSize)

    

def createErrorBarPlot(arr, xLabel, yLabel, xTicks, xTickLabels, ax = None, barColor = 'skyblue',offsetLabel = False, plotTopErrorOnly = True,plotSEM = True,rot = 45,
                       align = 'center', width = 0.8, colorTicks = True, ha = 'right',offsetTicks = False):

    """Create a Bar plot with error bars

    Args:
        arr: a 2D array of shape [N,P] showing N number datapoints for P parameters
        xLabel: label (str) of x axes
        yLabel: label (str) of y axes
        xTicks: values of ticks to plot (1d numeric arr)
        xTickLabels: labels to associate each tick to 
        ax: axes to plot on, can be None
        barColor: color of bar, any format that matplotlib can handle
    """
    # Create a DataFrame with random data, parameters are the columns, datapoints are the rows
    data = pd.DataFrame(arr, columns=[f'Variable {i+1}' for i in range(len(arr[0,:]))])

    # Calculate mean and standard deviation for each variable
    means = data.mean(axis = 0)

    if plotSEM:
        std_devs = data.std(axis = 0) / np.sqrt(11)
    else:
        std_devs = data.std(axis = 0)
    # Create a bar plot
    if ax == None:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
    
    if plotTopErrorOnly:
        # reformat errors to be 2 x N
        bottomErrors = np.zeros(std_devs.shape[0])
        yErrors = np.concatenate([bottomErrors.reshape(-1,1),np.asarray(std_devs).reshape(-1,1)], axis = 1).T
        ax.bar(means.index, means, yerr=yErrors, capsize=0, ecolor = barColor, color=barColor, alpha=0.7,align = align, width = width)
    else:
        ax.bar(means.index, means, yerr=std_devs, capsize=5, color=barColor, alpha=0.7)

    if offsetLabel:
        ax.set_xlabel(xLabel, labelpad=60)
    else:
        ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)

    
    
    if ha:
        ax.set_xticks(xTicks,labels = xTickLabels, rotation=rot, ha = ha)  # Rotate x-axis labels for better visibility
    else:
        ax.set_xticks(xTicks,labels = xTickLabels, rotation=rot)

    if offsetTicks == True:
        indices = np.arange(len(means))
        ax.set_xticks(indices+0.4)
        ax.set_xticklabels(xTickLabels, rotation=rot, ha=ha) 
    
    if colorTicks:
        for xtick, color in zip(ax.get_xticklabels(), barColor):
            xtick.set_color(color)

def createDoubleErrorBarPlot(arr1,arr2, xLabel, yLabel, xTicks, xTickLabels, ax = None, barColor = 'skyblue',offsetLabel = False, plotTopErrorOnly = True,plotSEM = True,rot = 45,
                       align = 'center', width = 0.8, colorTicks = True, ha = 'right'):

    """Create a Bar plot with error bars contrasting a change in arr 1 and arr 2

    Args:
        arr1: a 2D array of shape [N1,P] showing N number datapoints for P parameters
        arr2:a 2D array of shape [N2,P] showing N number datapoints for P parameters
        xLabel: label (str) of x axes
        yLabel: label (str) of y axes
        xTicks: values of ticks to plot (1d numeric arr)
        xTickLabels: labels to associate each tick to 
        ax: axes to plot on, can be None
        barColor: color of bar, any format that matplotlib can handle
    """
    # Create a DataFrame with random data, parameters are the columns, datapoints are the rows
    data1 = pd.DataFrame(arr1, columns=[f'Variable {i+1}' for i in range(len(arr1[0,:]))])
    data2 = pd.DataFrame(arr2, columns=[f'Variable {i+1}' for i in range(len(arr2[0,:]))])

    # Calculate mean and standard deviation for each variable
    means1 = data1.mean(axis = 0)
    means2 = data2.mean(axis = 0)

    if plotSEM:
        std_devs1 = data1.std(axis = 0) / np.sqrt(11)
        std_devs2 = data2.std(axis = 0) / np.sqrt(11)
    else:
        std_devs1 = data1.std(axis = 0)
        std_devs2 = data2.std(axis = 0)
    # Create a bar plot
    if ax == None:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
    
    if plotTopErrorOnly:
        # reformat errors to be 2 x N
        bottomErrors1 = np.zeros(std_devs1.shape[0])
        bottomErrors2 = np.zeros(std_devs2.shape[0])

        yErrors1 = np.concatenate([bottomErrors1.reshape(-1,1),np.asarray(std_devs1).reshape(-1,1)], axis = 1).T
        yErrors2 = np.concatenate([bottomErrors1.reshape(-1,1),np.asarray(std_devs1).reshape(-1,1)], axis = 1).T
        
         
        ax.bar(means.index, means, yerr=yErrors, capsize=0, ecolor = barColor, color=barColor, alpha=0.7,align = align, width = width)
    else:
        ax.bar(means.index, means, yerr=std_devs, capsize=5, color=barColor, alpha=0.7)

    if offsetLabel:
        ax.set_xlabel(xLabel, labelpad=60)
    else:
        ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    
    if ha:
        ax.set_xticks(xTicks,labels = xTickLabels, rotation=rot, ha = ha)  # Rotate x-axis labels for better visibility
    else:
        ax.set_xticks(xTicks,labels = xTickLabels, rotation=rot)

    if colorTicks:
        for xtick, color in zip(ax.get_xticklabels(), barColor):
            xtick.set_color(color)