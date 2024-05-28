
import numpy as np
import matplotlib.pyplot as plt

# Define the stages of request processing
stages = ['Optitrack tracking', 'Streaming', "Transmission" ,'Data pre-processing','Game engine refresh', 'Experimental dynamics', 'Display','Overall - worst case','Overall - measured']
# Define the average times for each stage
average_times = [5, 9, 1, 3, 25, 4, 17,64,47]  # Ensure 'Total' is the correct cumulative time
# Define the minimum and maximum times for each stage to create error bars
min_times = [3, 0, 1, 3, 0, 4, 17,30,50]
max_times = [7, 9, 1, 3, 25, 4, 17,64,43]

# Calculate the lower and upper error margins
lower_errors = [avg - min_time for avg, min_time in zip(average_times, min_times)]
upper_errors = [max_time - avg for max_time, avg in zip(max_times, average_times)]
error_bars = [lower_errors[-1], upper_errors[-1]]
# Calculate the cumulative start times for each bar, excluding the 'Total'
starts = [0] * len(average_times)
for i in range(1, len(average_times)):
    starts[i] = starts[i-1] + average_times[i-1]

# 'Total' starts at zero explicitly (can be adjusted if needed)

starts[-1] = 0
starts[-2] = 0

# Reverse the data so 'Total' is at the bottom and others stack downwards
stages.reverse()
average_times.reverse()
starts.reverse()


# Create list of colors
blueColor = (11/255, 201/255, 205/255)
redColor = (214/255, 50/255, 48/255)
orangeColor = (242/255, 197/255, 124/255)
poshBlackColor = (114/255, 9/255, 183/255)
colors = [(0.5,0.5,0.5),blueColor, redColor, orangeColor, poshBlackColor,(208/255,229/255,98/255),(178/255,121/255,167/255),(150/255,201/255,220/255),(217/255,202/255,179/255)]

fontsize = 18
# Create a horizontal bar chart with error bars
fig = plt.figure(figsize=(10, 5))
bars = plt.barh(stages[0], average_times[0], left=starts[0],color = colors[0],xerr = np.asarray(error_bars).reshape(2,1))
bars = plt.barh(stages[1], average_times[1], left=starts[1], color=colors[1])
bars = plt.barh(stages[2:], average_times[2:], left=starts[2:], color=colors[2:])

plt.xlabel('Latency (ms)', fontsize = fontsize-2)
#plt.ylabel("System Component", fontsize = fontsize)
#plt.title('System latency Breakdown', fontsize = fontsize)
plt.tick_params(labelsize = fontsize-4)
#plt.xlim(0, starts[0] + average_times[0] + max(upper_errors))  # Adjust x-axis to include error bars

# # Annotate the ends of the bars with the total time at that point
# for i, bar in enumerate(bars):
#     # Use i to get the corresponding stage and error correctly
#     plt.text(starts[i] + average_times[i] + upper_errors[i], bar.get_y() + bar.get_height()/2, f'{starts[i] + average_times[i]} ms',
#              va='center', ha='right', color='black', fontweight='bold')
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# plt.plot([100]* 6,[0,1,2,3,4,5], color = 'red')
fig.tight_layout()
# Display the plot
for label in ax.get_yticklabels():
    if label.get_text() in ['Overall - worst case','Overall - measured']:  # Make 'DNS Lookup' label bold
        label.set_fontweight('bold')
plt.savefig("latencyCascade.pdf")
plt.show()
