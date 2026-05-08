V8.3 changed/new files only.

Apply these files over your latest V8.2 project, preserving the folder structure.
Then rerun the notebook from the beginning with:

FORCE_REPROCESS_DYNAMIC = True
CNN_CALIBRATION_METHOD = "platt"
EDR_POLICY_MODE = "balanced"

New terminal demo:
python demo_realtime_edr.py --scenario notepad
python demo_realtime_edr.py --scenario ransomware
python demo_realtime_edr.py --mode replay --csv data/behavioral_raw/data/benign_021.csv
