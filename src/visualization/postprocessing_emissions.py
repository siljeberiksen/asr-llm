import pandas as pd


def runEmissionPostProcessing(experiment_name):
    # Load the CSV file into a DataFrame
    df = pd.read_csv("emissions.csv")
    # Display the first few rows
    results = df.project_name.apply(lambda a: experiment_name in a)
    df_filtered = df[results]
    count = 0
    for row in df_filtered["run_id"]:
        try:
            print(row)
            with open(f'emission_data/emissions_{row}.csv', "r", encoding="utf-8") as file:
                content = file.read()  # Read the entire file
                print(content)
            count += 1
        except:
            content = 0
       
    print(count)
    print("Duration", sum(df_filtered["duration"]))
    print("Emission", sum(df_filtered["emissions"]))
    print("CPU energy", sum(df_filtered["cpu_energy"]))
    print("GPU energy", sum(df_filtered["gpu_energy"]))
    print("RAM energy", sum(df_filtered["ram_energy"]))
