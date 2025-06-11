import time
import os
import sys
import traceback
from codecarbon import EmissionsTracker

tracker = EmissionsTracker(project_name="experiment_7")

try:
        from asr.asr_model_initialization import initialize_Whisper_model
        from experiments.experiment_7.proposed_system_experiment import run_experiment
        print("Running the script...")
        
        whisper_model = initialize_Whisper_model()   
        run_experiment('../result/npsc_samtale_experiment_7_llm.json', '../result/beam_npsc_experiment_7_llm.json',"../result/wer_npsc_experiment_7_llm.json", whisper_model, 10, tracker)
        print("Script completed successfully.")

except Exception as e:
    print("Caught an exception!")
    print(e)
    print(traceback.format_exc())
        
    tracker.stop()
    time.sleep(2)
        
    print("Restarting the script...")
    os.execv(sys.executable, [sys.executable, "-m", "experiments.handling_error_python2"])
except BaseException as e:
    print("Caught a BaseException!")
    print(e)
    print(traceback.format_exc())