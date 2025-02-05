from matplotlib import ticker
import matplotlib.pyplot as plt

# Data for the line chart
x_labels = ["0", "10", "50", "100"]  # Categorical x-axis labels
WER = [14.8, 75 ,36.5, 30.4]
CER = [6.7, 71 ,26.5, 21.9]


WER_MEDIAN = [8.1,11,9.5,10]
CER_MEDIAN = [2.6,3.96,3.3,3.75]

# Create positions for equal spacing
positions = range(len(x_labels))

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(positions, WER, label="WER", marker='o')  # WER line
plt.plot(positions, CER, label="CER", marker='s')  # CER line

# Set x-axis ticks with categorical labels
plt.xticks(ticks=positions, labels=x_labels)

# Set y-axis limits from 0 to 100
plt.ylim(0, 100)

# Add titles and labels
plt.title("Error Rate vs. History Length")
plt.xlabel("History Length")
plt.ylabel("Error Rates (%)")

# Format y-axis labels to show percentages
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

# Add grid and legend
plt.grid(True)
plt.legend()

# Show the plot
plt.show()