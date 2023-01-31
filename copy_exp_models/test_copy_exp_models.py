import pytest
import os
import copy_exp_models

# This tests copy_exp_model: are the folders duplicated
# in the target directory in the right structure?
# Are the files copied with their content?

# Make the from and to paths (as fixtures)
@pytest.fixture(autouse=True)
def from_path():
    from_path = os.path.abspath("./from_folder_test")
    return from_path

@pytest.fixture(autouse=True)
def to_path():
    to_path = os.path.abspath("./to_folder_test")
    return to_path

@pytest.fixture(autouse=True)
def from_path_subpaths(from_path):

    # Make the from directories first
    years = ["2023"]
    months = ["01"]
    days = []
    days.append(str(day) for day in range(10, 21))
    runs = []
    days.append("Run%s" %run for run in range(10, 21))
    models =  ["model_%s" %mod for mod in ["critic", "actor"]]
    from_path_subpaths = \
        [os.path.join(from_path, year, month, day, run, model) \
            for year in years for month in months for day in days \
                for run in runs for model in models]
    for p in from_path_subpaths:
        os.makedirs(p, exist_ok=True)
        with open(os.path.join(p, "test.txt", "w")) as f:
            f.write(p)

    return from_path_subpaths


def test_copy_models(from_path, to_path, from_path_subpaths):

    # Execute the command
    copy_exp_models.copy_models(from_path, to_path)

    # No go through all from_path_subpaths and check, if they exist
    # in to_path and have the right contend
    for p in from_path_subpaths:

        # Determine the corresponding folder in to_path
        corr_to_path = os.path.join(to_path, p.split(os.sep)[-5:0])

        # Assert that the folder and the text file with the
        # right content exist
        assert(os.path.isdir(corr_to_path), "directory was not corretly made")
        assert(os.path.isfile(os.path.join(corr_to_path, "test.txt")))
        with open(os.path.join(corr_to_path, "test.txt")) as f:
            assert(f.read() == p, "Wrong file content")
