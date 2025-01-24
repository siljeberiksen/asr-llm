import matplotlib.pyplot as plt

# Data for the line chart
x_labels = ["10", "50", "100", "1000"]  # Categorical x-axis labels
WER = [30, 40, 50, 60]
CER = [25, 35, 45, 44]

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