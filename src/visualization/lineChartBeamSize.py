import matplotlib.pyplot as plt

# Data for the line chart
x_labels = ["3", "5", "10", "20"]  # Categorical x-axis labels
WER = [30, 40, 50, 60]
CER = [25, 35, 45, 44]

WER_Prop = [10, 20, 30, 40]
CER_prop = [5, 15, 25, 35]

# Create positions for equal spacing
positions = range(len(x_labels))

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(positions, WER, label="WER baseline", marker='o', color="blue")  # WER line
plt.plot(positions, CER, label="CER baseline", marker='s', color = "blue")  # CER line

plt.plot(positions, WER_Prop, label="WER propose", marker='o', color = "red")  # WER line
plt.plot(positions, CER_prop, label="CER proposed", marker='s', color = "red")  # CER line

# Set x-axis ticks with categorical labels
plt.xticks(ticks=positions, labels=x_labels)

# Add titles and labels
plt.title("Error Rate vs. History Length")
plt.xlabel("Beam size")
plt.ylabel("Error Rate")

# Add grid and legend
plt.grid(True)
plt.legend()

# Show the plot
plt.show()