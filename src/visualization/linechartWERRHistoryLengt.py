from matplotlib import ticker
import matplotlib.pyplot as plt

# Data for the line chart
x_labels = ["10", "50", "100"]  # Categorical x-axis labels
WER = [-20.10, -12.98 , -7.90]
CER = [-34.50, -29.54, -15.00]


WER_MEDIAN = [-26.00,-17.65,-15.00]
CER_MEDIAN = [-35.29,-23.78,-18.33]

WER_no_filter = [-160.95, -669.86 , -1367.17]
CER_no_filter = [-286.05, -1364.15, -2726.03]


WER_MEDIAN_no_filter = [-138.01,-448.15,-1133.33]
CER_MEDIAN_no_filter = [-173.66,-731.11,-2869.16]

###Experiment 5-7
# WER = [14.8, 75 ,36.5, 30.4]
# CER = [6.7, 71 ,26.5, 21.9]
#WER_MEDIAN = [8.1,11,9.5,10]
#CER_MEDIAN = [2.6,3.96,3.3,3.75]
##Filtering out WER is worse than any beam
# WER = [14.8, 14.8 , 14.9 , 14.8]
# CER = [6.7, 6.6 , 6.7, 6.7]
# WER_MEDIAN = [8.1,7.6,7.8,7.6]
# CER_MEDIAN = [2.6,2.5,2.5,2.6]

##Filtering out where beams are equyal
# WER = [17.4, 17.2 , 17.3 , 17.2]
# CER = [7.6, 7.8 , 7.8, 7.8]
# WER_MEDIAN = [10.5,10.3,10.5,10.3]
# CER_MEDIAN = [3.4,3.4,3.4,3.4]

####Experiment 9-11

# Create positions for equal spacing
positions = range(len(x_labels))

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(positions, WER_MEDIAN, label="WERR mean", marker='o', color='tab:blue', linestyle='--')  # WER line
plt.plot(positions, CER_MEDIAN, label="CERR mean", marker='s',  color='tab:blue', linestyle='-')  # CER line
plt.plot(positions, WER, label="WERR median", marker='o', color='tab:orange', linestyle='--')  # WER line
plt.plot(positions, CER, label="CERR median", marker='s', color='tab:orange', linestyle='-')  # CER line

# Function to add value labels *above* each point
def add_labels(values, x_pos, offset=100):
    for x, y in zip(x_pos, values):
        plt.text(x, y - offset, f"{y:.1f}", ha='center', va='top', fontsize=9)

# Add labels
add_labels(WER, positions)
add_labels(CER, positions)
add_labels(WER_MEDIAN, positions)
add_labels(CER_MEDIAN, positions)
# Set x-axis ticks with categorical labels
plt.xticks(ticks=positions, labels=x_labels)

# Set y-axis limits from 0 to 100
#
plt.ylim(top=0)

# Add titles and labels
plt.title("")
plt.xlabel("History Length")
plt.ylabel("WERR/CERR (%)")

# Format y-axis labels to show percentages
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

# Add grid and legend
plt.grid(True)
plt.legend()

# Show the plot
plt.show()

