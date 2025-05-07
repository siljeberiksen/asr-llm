import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data based on WERR Mean values from your table
data = {
    "10": [-19.63, -20.15, -23.94, -26.51],
    "50": [-20.07, -14.31, -20.66, -24.37],
    "100": [-21.53, -14.69, -26.11, -23.30]
}

cerr_data = {
    "10": [-28.08, -29.32, -33.39, -43.09],
    "50": [-47.88, -23.15, -29.73, -42.35],
    "100": [-41.72, -20.01, -33.93, -43.80]  # Note: one positive outlier
}
beam_sizes = [3, 5, 10, 15]

df = pd.DataFrame(data, index=[f"Beam {b}" for b in beam_sizes])
df_clean = df.astype(float)

plt.figure(figsize=(8, 5))
sns.heatmap(df_clean, annot=True, fmt=".2f", cmap="RdYlGn", linewidths=0.5, cbar_kws={'label': 'Mean WERR (%)'})
plt.title("")
plt.xlabel("History Length")
plt.ylabel("Beam Size")

for text in plt.gca().texts:
    text.set_text(f"{text.get_text()}%")
plt.tight_layout()
plt.show()