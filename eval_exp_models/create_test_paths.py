'''
This creates a test dir tree to test copy_exp_models on
'''
import os
import json
from pathlib import Path

def from_path_subpaths(root_path):
    '''
    Creates some folders and content for testing
    copy_models on
    '''

    # Make the from directories first
    years = ["2023"]
    months = ["01"]
    days = [str(day) for day in range(10, 21)]
    runs = ["Run%s" %run for run in range(10, 21)]
    models =  ["model_%s" %mod for mod in ["critic", "actor"]]
    subpaths = \
        [os.path.join(root_path, year, month, day, run, model) \
            for year in years for month in months for day in days \
                for run in runs for model in models]
    for p in subpaths:
        os.makedirs(p, exist_ok=True)
        with open(os.path.join(p, "test.txt"), "w") as f:
            f.write(p)
        with (Path(p).parent/"parameters.json").open(mode="w") as par_file:
            json.dump({"test": "test"}, par_file, ensure_ascii=False, indent=4)

    return subpaths


if __name__ == "__main__":
    test_path = os.path.abspath("./from_folder_test")
    from_path_subpaths(root_path=test_path)