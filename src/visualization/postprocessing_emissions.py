import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv("emissions.csv")

# Display the first few rows
results = df.project_name.apply(lambda a: "experiment_3" in a)
df_filtered = df[results]

print(df_filtered)
print(df)
