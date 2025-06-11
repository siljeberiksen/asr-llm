import pandas as pd


def runEmissionPostProcessing(experiment_name, emission_file = "emissions.csv"):
    df = pd.read_csv(emission_file)
    results = df.project_name.apply(lambda a: experiment_name in a)
    df_filtered = df[results]
    count = 0
    for row in df_filtered["run_id"]:
        try:
            with open(f'emission_data/emissions_{row}.csv', "r", encoding="utf-8") as file:
                content = file.read()  # Read the entire file
            count += 1
        except:
            content = 0
  
    numeric_columns = ["duration", "emissions", "cpu_energy", "gpu_energy", "ram_energy"]
    for col in numeric_columns:
        df_filtered[col] = pd.to_numeric(df_filtered[col], errors="coerce")

    invalid_rows = df_filtered[df_filtered[numeric_columns].isna().any(axis=1)]
    if not invalid_rows.empty:
        print("Invalid numeric values found in the following rows:")
        print(invalid_rows)   
    print(count)
    print("Duration", sum(df_filtered["duration"]))
    print("Emission", sum(df_filtered["emissions"]))
    print("CPU energy", sum(df_filtered["cpu_energy"]))
    print("GPU energy", sum(df_filtered["gpu_energy"]))
    print("RAM energy", sum(df_filtered["ram_energy"]))
