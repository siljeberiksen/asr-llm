from matplotlib import ticker
import matplotlib.pyplot as plt

# Data for the line chart
x_labels = ["10", "50", "100"]  # Categorical x-axis labels
WERR_gemma = [-0.86, 0.09 , 1.33]
#CER = [-29.32, -23.15, -20.01]


WERR_LLama = [-30.88,-21.92,-14.60]
#CER_10 = [-35.29,-23.78,-18.33]

# WER_no_filter = [-160.95, -669.86 , -1367.17]
# CER_no_filter = [-286.05, -1364.15, -2726.03]


# WER_MEDIAN_no_filter = [-138.01,-448.15,-1133.33]
# CER_MEDIAN_no_filter = [-173.66,-731.11,-2869.16]

# Create positions for equal spacing
positions = range(len(x_labels))

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(positions, WERR_gemma, label="Gemma", marker='o', color='tab:red', linestyle='-')  # WER line
plt.plot(positions, WERR_LLama, label="LLaMA", marker='o',  color='tab:blue', linestyle='-')  # CER line
#plt.plot(positions, WER, label="WERR median", marker='o', color='tab:orange', linestyle='--')  # WER line
#plt.plot(positions, CER, label="CERR median", marker='s', color='tab:orange', linestyle='-')  # CER line

# Set x-axis ticks with categorical labels
plt.xticks(ticks=positions, labels=x_labels)

# Set y-axis limits from 0 to 100
#


# Add titles and labels
plt.title("")
plt.xlabel("History Length")
plt.ylabel("WERR (%)")

# Format y-axis labels to show percentages
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

# Add grid and legend
plt.grid(True)
plt.legend(title="LLM model")

# Show the plot
plt.show()
