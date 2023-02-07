'''
This tests create_test_paths
'''
import pytest
import os
import create_test_paths

@pytest.fixture(autouse=True, name="from_path")
def make_from_path():
    model_path = os.path.abspath("./from_folder_test")
    return model_path

def test_from_path_subpaths(from_path):
    '''
    Lets create_test_paths create test paths
    and then checks if they are all there with
    the right content
    '''

    # Make the test_paths
    create_test_paths.from_path_subpaths(root_path=from_path)

    # Make sure all directories are there
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
        assert os.path.isdir(p), f"Path {p} not found"
        with open(os.path.join(p, "test.txt"), "r") as f:
            assert f.read() == p, f"Wrong file content in {p}"