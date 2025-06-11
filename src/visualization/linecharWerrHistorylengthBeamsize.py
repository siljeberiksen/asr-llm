from matplotlib import ticker
import matplotlib.pyplot as plt

x_labels = ["10", "50", "100"]  # Categorical x-axis labels
WER = [-20.15, -14.31 , -14.69]
#CER = [-29.32, -23.15, -20.01]


WER_10 = [-23.94,-20.65,-26.11]
#CER_10 = [-35.29,-23.78,-18.33]

# WER_no_filter = [-160.95, -669.86 , -1367.17]
# CER_no_filter = [-286.05, -1364.15, -2726.03]


# WER_MEDIAN_no_filter = [-138.01,-448.15,-1133.33]
# CER_MEDIAN_no_filter = [-173.66,-731.11,-2869.16]

# Create positions for equal spacing
positions = range(len(x_labels))

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(positions, WER, label="5", marker='o', color='tab:red', linestyle='-')  # WER line
plt.plot(positions, WER_10, label="10", marker='o',  color='tab:blue', linestyle='-')  # CER line
#plt.plot(positions, WER, label="WERR median", marker='o', color='tab:orange', linestyle='--')  # WER line
#plt.plot(positions, CER, label="CERR median", marker='s', color='tab:orange', linestyle='-')  # CER line

plt.xticks(ticks=positions, labels=x_labels)
plt.ylim(top=0)

plt.title("")
plt.xlabel("History Length")
plt.ylabel("WERR (%)")

plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

plt.grid(True)
plt.legend(title="Beam size")

plt.show()
