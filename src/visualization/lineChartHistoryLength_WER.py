import matplotlib.pyplot as plt

# Data for the line chart
x_labels = ["0", "10", "100"]  # Categorical x-axis labels
WER = [14.8, 75 , 30.4]
CER = [6.7, 71 , 21.9]

# Create positions for equal spacing
positions = range(len(x_labels))

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(positions, WER, label="WER", marker='o')  # WER line
plt.plot(positions, CER, label="CER", marker='s')  # CER line

# Set x-axis ticks with categorical labels
plt.xticks(ticks=positions, labels=x_labels)

# Add titles and labels
plt.title("Error Rate vs. History Length")
plt.xlabel("History Length")
plt.ylabel("Error Rate")

# Add grid and legend
plt.grid(True)
plt.legend()

# Show the plot
plt.show()