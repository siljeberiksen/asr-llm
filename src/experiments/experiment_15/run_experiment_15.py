import time
import os
import sys
import traceback
from codecarbon import EmissionsTracker

tracker = EmissionsTracker(project_name="experiment_15")


COUNT_FILE = "experiments.experiment_13.count.txt"

def load_count():
    if os.path.exists(COUNT_FILE):
        try:
            with open(COUNT_FILE, "r") as f:
                return int(f.read().strip())
        except ValueError:
            return 0  # Default to 0 if file contents are invalid
    return 0
# Function to save count to file
def save_count(count):
    with open(COUNT_FILE, "w") as f:
        f.write(str(count))

count = load_count()

try:
        from asr.asr_model_initialization import initialize_Whisper_model
        from experiments.experiment_15.proposed_system_experiment import run_experiment

        # Your main script logic here
        print("Running the script...")
        
        # Run experiment
        whisper_model = initialize_Whisper_model()   
        run_experiment('../result/npsc_samtale_experiment_15_llm.json', '../result/beam_npsc_experiment_15_llm.json',"../result/wer_npsc_experiment_15_llm.json", whisper_model, 10, tracker, count)
        # If no exception occurs, break the loop and finish
        print("Script completed successfully.")

except Exception as e:
    print("Caught an exception!")
    print(e)
    print(traceback.format_exc())
        
        # Optional: Add a delay before restarting
    tracker.stop()
    time.sleep(2)
        
        # Restart the script
    print("Restarting the script...")
    os.execv(sys.executable, [sys.executable, "-m", "experiments.experiment_15.run_experiment_15"])
except BaseException as e:
    print("Caught a BaseException!")
    print(e)
    print(traceback.format_exc())