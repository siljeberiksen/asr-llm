from matplotlib import ticker
import matplotlib.pyplot as plt

x_labels = ["0", "10", "50", "100"]  # Categorical x-axis labels
WER = [14.8, 75 ,36.5, 30.4]
CER = [6.7, 71 ,26.5, 21.9]


WER_MEDIAN = [8.1,11,9.5,10]
CER_MEDIAN = [2.6,3.96,3.3,3.75]

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
plt.plot(positions, WER_MEDIAN, label="WER", marker='o')  # WER line
plt.plot(positions, CER_MEDIAN, label="CER", marker='s')  # CER line
plt.xticks(ticks=positions, labels=x_labels)
plt.ylim(0, 100)

plt.title("Error Rate vs. History Length")
plt.xlabel("History Length")
plt.ylabel("Error Rates (%)")

plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

plt.grid(True)
plt.legend()

plt.show()