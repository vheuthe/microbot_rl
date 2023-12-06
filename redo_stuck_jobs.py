# Searches a directory for unfinished runs and reruns them
import sys
import os
from pathlib import Path


def redo_job(unfinished_dir):
    # Search the directory for folders that contain a training.h5
    # but not a evaluation.h5. If that is true, rerun the simulattion
    # in that dir
    for root_dir, _, files in os.walk(unfinished_dir):
        if "model_" in root_dir:
            continue
        elif "training.h5" in files and not "evaluation.h5" in files:
            os.system(f"python3 submit_dir.py {root_dir}")

if __name__ == "__main__":
    # Check, if there is input
    assert len(sys.argv) > 1, "You need to specify a directory"
    unfinished_dir = Path(sys.argv[1])

    # Now redo the job
    redo_job(unfinished_dir=unfinished_dir)