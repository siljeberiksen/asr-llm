import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data based on WERR Mean values from your table
data = {
    "10": [-1.68, -1.30, -2.69, -7.18],
    "50": [-6.43, -2.17, -5.57, -6.72],
    "100": [-8.81, -4.88, -8.73, -6.11]
}

cerr_data = {
    "10": [-16.35, -10.16, -12.15, -21.27],
    "50": [-27.77, -13.47, -17.17, -21.28],
    "100": [-31.36, -14.26, -20.48, -17.58]  # Note: one positive outlier
}
beam_sizes = [3, 5, 10, 15]

df = pd.DataFrame(cerr_data, index=[f"Beam {b}" for b in beam_sizes])
df_clean = df.astype(float)

plt.figure(figsize=(8, 5))
sns.heatmap(df_clean, annot=True, fmt=".2f", cmap="RdYlGn", linewidths=0.5, cbar_kws={'label': 'Mean WERR (%)'})
plt.title("")
plt.xlabel("History Length (l)")
plt.ylabel("Beam Size (k)")

for text in plt.gca().texts:
    text.set_text(f"{text.get_text()}%")
plt.tight_layout()
plt.show()