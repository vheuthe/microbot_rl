import os

# This creates a test dir tree to test copy_exp_models on

def from_path_subpaths(from_path):

    # Make the from directories first
    years = ["2023"]
    months = ["01"]
    days = [str(day) for day in range(10, 21)]
    runs = ["Run%s" %run for run in range(10, 21)]
    models =  ["model_%s" %mod for mod in ["critic", "actor"]]
    from_path_subpaths = \
        [os.path.join(from_path, year, month, day, run, model) \
            for year in years for month in months for day in days \
                for run in runs for model in models]
    for p in from_path_subpaths:
        os.makedirs(p, exist_ok=True)
        with open(os.path.join(p, "test.txt"), "w") as f:
            f.write(p)

    return from_path_subpaths


if __name__ == "__main__":
    from_path = os.path.abspath("./from_folder_test")
    from_path_subpaths(from_path=from_path)