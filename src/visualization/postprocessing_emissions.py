import pandas as pd


def runEmissionPostProcessing(experiment_name):
    # Load the CSV file into a DataFrame
    df = pd.read_csv("emissions.csv")
    # Display the first few rows
    results = df.project_name.apply(lambda a: experiment_name in a)
    df_filtered = df[results]

    print(df_filtered)
    print(df)
    print(df.columns)
    print(df["run_id"])
    for row in df["run_id"]:
        try:
            with open(f'./src/emission_data/emission_base_{row}', "r", encoding="utf-8") as file:
                content = file.read()  # Read the entire file
            print("content", content)
        except:
            print("NOOOOO")
        print(row)
    print("Duration", sum(df["duration"]))
    print("Emission", sum(df["emissions"]))
    print("CPU energy", sum(df["cpu_energy"]))
    print("GPU energy", sum(df["gpu_energy"]))
    print("RAM energy", sum(df["ram_energy"]))
    print(df.columns)
