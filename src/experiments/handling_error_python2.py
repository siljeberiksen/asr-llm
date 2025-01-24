import time
import os
import sys

from asr.asr_model_initialization import initialize_Whisper_model
from experiments.experiment_5.proposed_system_experiment import run_experiment


while True:
    try:
        # Your main script logic here
        print("Running the script...")
        
        # Run experiment
        whisper_model = initialize_Whisper_model()   
        run_experiment('../result/npsc_samtale_experiment_5_llm.json', '../result/beam_npsc_experiment_5_llm.json',"../result/wer_npsc_experiment_5_llm.json", whisper_model)
        # If no exception occurs, break the loop and finish
        print("Script completed successfully.")
        break 

    except:

        
        # Optional: Add a delay before restarting
        time.sleep(2)
        
        # Restart the script
        print("Restarting the script...")
        os.execv(sys.executable, [sys.executable, "-m", "experiments.handling_error_python2"])