import time
import os
import sys
import traceback
from codecarbon import EmissionsTracker

tracker = EmissionsTracker(project_name="experiment_9")

try:
        from asr.asr_model_initialization import initialize_Whisper_model
        from experiments.experiment_9.proposed_system_experiment import run_experiment

        # Your main script logic here
        print("Running the script...")
        
        # Run experiment
        whisper_model = initialize_Whisper_model()   
        run_experiment('../result/npsc_samtale_experiment_9_llm.json', '../result/beam_npsc_experiment_9_llm.json',"../result/wer_npsc_experiment_9_llm.json", whisper_model, 10, tracker)
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
    os.execv(sys.executable, [sys.executable, "-m", "experiments.experiment_9.run_experiment_9"])
except BaseException as e:
    print("Caught a BaseException!")
    print(e)
    print(traceback.format_exc())