"""
This probes the nature of uninstructed movements
"""


# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import sys

# add current path to system PATH 
sys.path.insert(0,'/Users/ashwin/Documents/Y4 project Brain Human Interfaces/General 4th year Github repo/BrainMachineInterfaceDataAnalysis')

# Import user defined libraries
import DataExtraction.extractRawData as dataExtractor
from BasicAnalytics import targetAcqusitionPlotting as targetPlotter
from BasicAnalytics import variabilityAnalysis 

print(os.getcwd())
# Fetch key training data for variability analysis
rigidBodyTrain1, scores, noParticipants = dataExtractor.retrieveTrainingData()



# Calculate variability values of training data
rms_values = variabilityAnalysis.calculateVariabilityScores(rigidBodyTrain1)
print(rms_values.shape)

# Calculate variability values of individual body parts by summing over dof for each body part
rmsValuesRigidBodyParts = np.sum(rms_values.reshape(19,3,noParticipants*5),axis = 1).reshape(19,noParticipants*5)
print(rms_values)


rms_values = rmsValuesRigidBodyParts

# Score r values 
maxVal = np.max(rms_values)
rms_values = rms_values / maxVal

x = range(0, 19)
labels = ['PVS', 'AB', 'CH', 'NCK', 'HD', 'LSD', 'LUA', 
'LFA', 'LHD', 'RSD', 'RUA', 'RFA', 'RHD',  'LTH', 'LSN', 
'LFT', 'RTH', 'RSN', 'RFT']
  






def plotScatterPlotVariability(scores):

    # Average score over trials
    scores = np.average(scores,axis = 0)

    # Define x axis on plots
    x = range(0, 19)
    plt.scatter(x,np.average(rms_values[:,0:noParticipants],axis = 1), label = "Trial 1: Avg Score {}".format(round(scores[0])))
    plt.scatter(x,np.average(rms_values[:,noParticipants:noParticipants*2],axis = 1), label = "Trial 2: Avg Score {}".format(round(scores[1])))
    plt.scatter(x,np.average(rms_values[:,noParticipants*2:noParticipants*3],axis = 1), label = "Trial 3: Avg Score {}".format(round(scores[2])))
    plt.scatter(x,np.average(rms_values[:,noParticipants*3:noParticipants*4],axis = 1), label = "Trial 4: Avg Score {}".format(round(scores[3])))
    plt.scatter(x,np.average(rms_values[:,noParticipants*4:noParticipants*5],axis = 1), label = "Trial 5: Avg Score {}".format(round(scores[4])))

    # Sample data

    labels = ['PVS', 'AB', 'CH', 'NCK', 'HD', 'LSD', 'LUA', 
                        'LFA', 'LHD', 'RSD', 'RUA', 'RFA', 'RHD',  'LTH', 'LSN', 
                        'LFT', 'RTH', 'RSN', 'RFT']

    # Set custom x-tick labels
    plt.xticks(x, labels)
    plt.xlabel("Body Parts", fontsize = 20)
    plt.ylabel("Average variability", fontsize = 20)
    plt.title("Average variability metrics for all body parts for each trial", fontsize = 20)
    plt.legend(fontsize = 15)
    plt.tick_params(axis='y', which='major', labelsize=14)
    plt.tick_params(axis='x', which='major', labelsize=14)
    plt.show()
plotScatterPlotVariability(rms_values)

# Plot box plots ---

# Normalisation and sum variability values across DOF
summedRMSvaluesAcrossDOF = np.sum(rms_values.reshape(19,5,noParticipants),axis = 0)
summedRMSvaluesAcrossDOF = summedRMSvaluesAcrossDOF / np.max(summedRMSvaluesAcrossDOF)




red = (245/255,5/255,5/255) # (RGB) or F50505 (Hex)


def createBoxPlot(ax,listOfVars,colorList,xTickList,xlabel,ylabel):
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

    ax.set_xlabel(xlabel,fontsize = 22,fontweight='bold')
    ax.set_ylabel(ylabel,fontsize = 22,fontweight='bold')
    ax.tick_params(labelsize = 20)
    ax.set_xticklabels(xTickList)

    # Remove top and right spines for the first plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

colors = [red, red, red, red, red]
xTickList = ['Trial 1', 'Trial 2', 'Trial 3', 'Trial 4', 'Trial 5']
trialSplitSummedRMSVals = [summedRMSvaluesAcrossDOF[0,:], summedRMSvaluesAcrossDOF[1,:], summedRMSvaluesAcrossDOF[2,:], summedRMSvaluesAcrossDOF[3,:],summedRMSvaluesAcrossDOF[4,:]]

fig= plt.figure(figsize=(10,6))
ax = plt.gca()
createBoxPlot(ax, trialSplitSummedRMSVals, colors, xTickList, 'Trial No', 'Variability metric')
plt.title("Box plot of variability of rigid bodies across trials \n with line of average variability", fontsize = 20)
# Middle of box plot is median and line is average so there may be differences
plt.plot(np.linspace(1,5,5),np.average(summedRMSvaluesAcrossDOF,axis = 1),label = "Average variability")
plt.legend(fontsize = 15)
fig.tight_layout() 
plt.show()


fig= plt.figure(figsize=(10,6))
ax = plt.gca()
scoresBoxPlot = [scores[:,0], scores[:,1], scores[:,2], scores[:,3], scores[:,4]]
createBoxPlot(ax, scoresBoxPlot, colors, xTickList, 'Trial No', 'Score')
plt.plot(np.linspace(1,5,5),np.average(scores,axis = 0),label = "Average variability")
plt.title("Box plot of scores achieved across trials \n with line to show average score", fontsize = 20)
fig.tight_layout() 
plt.legend(fontsize = 15)
plt.show()


fig= plt.figure(figsize=(12,6))
ax = plt.gca()
variabilityBoxPlot = []
for i in range(len(labels)):
    variabilityBoxPlot.append(rms_values[i,:].T)

# Normalise to 1

createBoxPlot(ax, variabilityBoxPlot, ['red']*len(labels), labels, 'Trial No', 'Average variability')
plt.scatter(np.linspace(1,19,19),np.average(rms_values,axis = 1),label = "Average Variability by body part")
plt.title("Box plot of average variability achieved across body parts \n with points to show average", fontsize = 20)
fig.tight_layout() 
plt.legend(fontsize = 15)
plt.show()

def get_color_hex(value, light_color, dark_color):
    """Function returns hexadecimal color based on value between 0 and 1."""
    # Check value is bounded first
    value = max(0, min(1, value))

    # Interpolate between light and dark colors
    r = int(light_color[0] + value * (dark_color[0] - light_color[0]))
    g = int(light_color[1] + value * (dark_color[1] - light_color[1]))
    b = int(light_color[2] + value * (dark_color[2] - light_color[2]))

    # Return as hexadecimal
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)


from sklearn.decomposition import PCA
def performPCAAndPlot(data, n_components=2):
    """
    Performs PCA on the given dataset to reduce its dimensionality and plots the first two components.
    
    Parameters:
    - data: A 2D numpy array with shape (observations, features).
    - n_components: The number of principal components to keep.
    
    Returns:
    - A 2D numpy array with the transformed dataset of shape (observations, n_components).
    - The PCA model for further inspection or transformation.
    """
    # Initialize the PCA model
    pca = PCA(n_components=n_components)
    
    # Fit the PCA model to the data and transform the data onto the principal components
    transformed_data = pca.fit_transform(data)
    
    # Plot the first two components
    plt.figure(figsize=(8, 6))
    plt.scatter(transformed_data[[0,10,20,30,40], 0], transformed_data[[0,10,20,30,40], 1], alpha=0.7, edgecolors='w',color='b', s=100)
    plt.scatter(transformed_data[[1,11,21,31,41], 0], transformed_data[[1,11,21,31,41], 1], alpha=0.7, edgecolors='w',color='r', s=100)
    plt.scatter(transformed_data[[2,12,22,32,42], 0], transformed_data[[2,12,22,32,42], 1], alpha=0.7, edgecolors='w',color='m', s=100)
    plt.scatter(transformed_data[[3,13,23,33,43], 0], transformed_data[[3,13,23,33,43], 1], alpha=0.7, edgecolors='w',color='g', s=100)
    plt.scatter(transformed_data[[4,14,24,34,44], 0], transformed_data[[4,14,24,34,44], 1], alpha=0.7, edgecolors='w',color='k', s=100)
    plt.scatter(transformed_data[[5,15,25,35,45], 0], transformed_data[[5,15,25,35,45], 1], alpha=0.7, edgecolors='w',color='teal', s=100)
    plt.scatter(transformed_data[[1,16,26,36,46], 0], transformed_data[[6,16,26,36,46], 1], alpha=0.7, edgecolors='w',color='cyan', s=100)
    plt.scatter(transformed_data[[2,17,27,37,47], 0], transformed_data[[7,17,27,37,47], 1], alpha=0.7, edgecolors='w',color='violet', s=100)
    plt.scatter(transformed_data[[3,18,28,38,48], 0], transformed_data[[8,18,28,38,48], 1], alpha=0.7, edgecolors='w',color='coral', s=100)
    plt.scatter(transformed_data[[4,19,29,39,49], 0], transformed_data[[9,19,29,39,49], 1], alpha=0.7, edgecolors='w',color='c', s=100)
    
    plt.title('Participants PCA variability metrics for all trials (participants are color coded)')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True)
    plt.show()

    # Color code according to score
    scores1D = scores.reshape(-1,order = 'F') # from 10 x 5 the F states that we traverse throigh the columns first
    maxScore = max(scores1D)



    # colour
    
    white =  (255, 255, 255) # White
    maxColor = (245,5,5) # (RGB) or F50505 (Hex)

    for i in range(len(transformed_data[:,0])):

        # Find score
        score = scores1D[i]

        # Find color
        color = get_color_hex(score/maxScore,white,maxColor)

        plt.scatter(transformed_data[i, 0], transformed_data[i, 1], alpha=1, edgecolors='w',color=color, s=100)
    
    
    plt.title('Participants PCA variability metrics for all trials (color coded by score)')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True)
    plt.show()

    return transformed_data, pca

transformedData,pca = performPCAAndPlot(rms_values.T)

from sklearn.cluster import KMeans

def kMeansClustering(data, n_clusters):
    """
    Performs k-means clustering on m-dimensional data.
    
    Parameters:
    - data: A 2D numpy array of shape (num_samples, num_features) where num_samples is the number of observations
            and num_features is the dimensionality of each observation (m).
    - n_clusters: The number of clusters to form as well as the number of centroids to generate (n).
    
    Returns:
    - labels: A 1D numpy array of cluster labels for each observation.
    - centroids: A 2D numpy array representing the coordinates of each cluster center.
    """
    # Initialize the KMeans model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    
    # Fit the model to the data and predict the cluster labels
    labels = kmeans.fit_predict(data)
    
    # Extract the cluster centroids
    centroids = kmeans.cluster_centers_
    
    return labels, centroids

# Perform k-means clustering
labels, centroids = kMeansClustering(transformedData, 4)

print(labels,centroids)

plt.scatter(transformedData[:, 0], transformedData[:, 1], c=labels, s=50, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.5, marker='X') # Centroids
plt.title('K-Means Clustering of 2D Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()


def kMeansClusteringAndPlot(data, n_clusters_list):
    """
    Performs k-means clustering on the given dataset for a list of cluster numbers
    and plots each clustering result in a 2x2 subplot grid.
    
    Parameters:
    - data: A 2D numpy array of shape (num_samples, num_features).
    - n_clusters_list: A list of integers specifying the number of clusters to use in each subplot.
    """
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))  # Setup for 2x2 subplots
    axs = axs.ravel()  # Flatten the 2x2 array of axes for easy indexing
    
    labelList = []
    for i, n_clusters in enumerate(n_clusters_list):
        # Initialize and fit KMeans model
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(data)
        labelList.append(labels)
        centroids = kmeans.cluster_centers_
        
        # Plotting the results on the ith subplot
        axs[i].scatter(data[:, 0], data[:, 1], c=labels, s=50, cmap='viridis')
        axs[i].scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.5, marker='X')  # Centroids
        axs[i].set_title(f'{n_clusters} Clusters')
        axs[i].set_xlabel('Feature 1')
        axs[i].set_ylabel('Feature 2')

    plt.tight_layout()
    plt.show()

    return labelList,centroids


# List of cluster counts for each subplot
n_clusters_list = [2, 3, 4, 5]

# Perform k-means clustering and plot for 2, 3, 4, and 5 clusters
kMeansLabels, centroids = kMeansClusteringAndPlot(transformedData, n_clusters_list)

# Now we want to plot average variability metrics for each category
# We try the 4 cluster one
kmeansLabel4Cluster = kMeansLabels[2] # 3rd one


# Create a dictionary to hold the indices for each number
indices_dict = {0: [], 1: [], 2: [], 3: []}

# Iterate through the list and append the indices to the respective lists in the dictionary
for index, number in enumerate(kmeansLabel4Cluster):
    if number in indices_dict:
        indices_dict[number].append(index)

# Now average over each cluster

zero = np.average(rms_values[:,indices_dict[0]].T,axis = 0)
one = np.average(rms_values[:,indices_dict[1]].T,axis = 0)
two = np.average(rms_values[:,indices_dict[2]].T,axis = 0)
three = np.average(rms_values[:,indices_dict[3]].T,axis = 0)

plt.scatter(np.linspace(1,19,19),zero, label = "Cluster 1")
plt.scatter(np.linspace(1,19,19),one, label = "Cluster 2")
plt.scatter(np.linspace(1,19,19),two, label = "Cluster 3")
plt.scatter(np.linspace(1,19,19),three, label = "Cluster 4")

labels = ['PVS', 'AB', 'CH', 'NCK', 'HD', 'LSD', 'LUA', 
                      'LFA', 'LHD', 'RSD', 'RUA', 'RFA', 'RHD',  'LTH', 'LSN', 
                      'LFT', 'RTH', 'RSN', 'RFT']

# Set custom x-tick labels
plt.xticks(x, labels)
plt.xlabel("Body Parts", fontsize = 20)
plt.ylabel("Average variability", fontsize = 20)
plt.title("Average variability metrics for all body parts for each trial", fontsize = 20)
plt.legend(fontsize = 15)
plt.tick_params(axis='y', which='major', labelsize=14)
plt.tick_params(axis='x', which='major', labelsize=14)
plt.show()

