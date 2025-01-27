import time
import os
import sys
import traceback
from codecarbon import EmissionsTracker

tracker = EmissionsTracker(project_name="experiment_3")
try:
     # Wrap the imports in the same try block
    from experiments.experiment_3.proposed_system_experiment import run_experiment
    from experiments.experiment_3.proposed_system_experiment import initialize_Whisper_model
    # Your main script logic here
    print("Running the script...")
        
    # Run experiment
    whisper_model = initialize_Whisper_model()   
    run_experiment('../result/npsc_samtale_experiment_3_llm.json', '../result/beam_npsc_experiment_3_llm.json',"../result/wer_npsc_experiment_3_llm.json", whisper_model,  tracker =tracker)
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
    os.execv(sys.executable, [sys.executable, "-m", "experiments.handling_error_python"])

except BaseException as e:
    print("Caught a BaseException!")
    print(e)
    print(traceback.format_exc())