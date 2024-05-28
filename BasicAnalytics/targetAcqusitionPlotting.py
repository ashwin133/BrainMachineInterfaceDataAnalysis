"""
Contains code for plotting target acquisitions
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Define the corners of the rectangle
def plotSuccessfulTrajectoryToTarget(trialInformation,targetNo=0,cursorWidth = 30):
    """
    Plots a trajectory to a target on standard axes

    Args:
        trialInformation: Dict corresponding to trial output information 
        targetNo: index of target acquisition
        plotMiddlePath: path is plotted for middle of box rather than default bottom left
    """

    # Calc start and end timestamp indexes for first target acquisition
    startIdx = trialInformation['successfulGoCues'][targetNo]
    endIdx = trialInformation['successfulTargetReached'][targetNo] 

    # Plot first target acquisition
    plt.figure(figsize=(11,6))
    plt.plot(trialInformation['cursorPos'][startIdx:endIdx,0]+cursorWidth // 2,trialInformation['cursorPos'][startIdx:endIdx,1] + cursorWidth //2)

    # Set size of display similar to screen size
    plt.xlim(-20,1920)
    plt.ylim(-20,1045)

    plotTarget(trialInformation['successfulTargetBoxLocs'][targetNo])

    plotCursor(trialInformation['cursorPos'][startIdx,0:2],color = 'y')

    plotCursor(trialInformation['cursorPos'][endIdx-1,0:2],color = 'g')

def plotFailedTrajectoryToTarget(trialInformation,targetNo=0,cursorWidth = 30):
    """
    Plots a trajectory to a target on standard axes

    Args:
        trialInformation: Dict corresponding to trial output information 
        targetNo: index of target acquisition
    """

    

    # Calc start and end timestamp indexes for first target acquisition
    startIdx = trialInformation['failedGoCues'][targetNo]
    endIdx = trialInformation['failedTargetReached'][targetNo] - 1

    # Plot first target acquisition
    plt.figure(figsize=(11,6))
    plt.plot(trialInformation['cursorPos'][startIdx:endIdx,0] + cursorWidth // 2,trialInformation['cursorPos'][startIdx:endIdx,1] + cursorWidth // 2)

    # Set size of display similar to screen size
    plt.xlim(-20,1920)
    plt.ylim(-20,1045)

    plotTarget(trialInformation['failedTargetBoxLocs'][targetNo])

    plotCursor(trialInformation['cursorPos'][startIdx,0:2],color = 'y')

    plotCursor(trialInformation['cursorPos'][endIdx-1,0:2],color = 'g')

    

def plotTarget(boxLoc, width = 60):
    """
    Function to plot a target on standard plotting axes

    Args:
        boxLoc: list [x,y] corresponding to bottom left corner location
        width: int corresponding to width and height of box
    """
    # increase width by 10 as this is the approximate resolution of pygame
    tolerance = 10 
    x1, y1 = boxLoc[0] - tolerance , boxLoc[1] - tolerance // 2
    x2, y2 = x1 + width + tolerance , y1 + width + tolerance

    # Get current axis
    ax = plt.gca()

    # Create a rectangle patch
    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='k', facecolor='red', alpha=0.8)

    # Add the rectangle to the plot
    ax.add_patch(rect)

def plotCursor(cursorLoc, width = 35, color = 'r'):
    """
    Function to plot a cursor on standard plotting axes

    Args:
        cursorLoc: list [x,y] corresponding to bottom left cursor location
        width: int corresponding to width of cursor
    """

    tolerance = 10 
    x1, y1 = cursorLoc[0] - tolerance//2 , cursorLoc[1] - tolerance // 2
    x2, y2 = x1 + width + tolerance , y1 + width + tolerance

    # Get current axis
    ax = plt.gca()

    # Create a rectangle patch
    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='k', facecolor=color, alpha=0.8)

    # Add the rectangle to the plot
    ax.add_patch(rect)


