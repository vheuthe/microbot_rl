'''
This tests copy_exp_model: are the folders duplicated
in the target directory in the right structure?
Are the files copied with their content?
'''
import os
import pytest
import copy_exp_models
import create_test_paths

# Make the from and to paths (as fixtures)
@pytest.fixture(autouse=True, name="from_path")
def make_from_path():
    source_path = os.path.abspath("./from_folder_test")
    return source_path

@pytest.fixture(autouse=True, name="to_path")
def make_to_path():
    dest_path = os.path.abspath("./to_folder_test")
    return dest_path

@pytest.fixture(autouse=True, name="from_path_subpaths")
def make_subpaths(from_path):

    # Make the from directories first
    subpaths = create_test_paths.from_path_subpaths(root_path=from_path)

    return subpaths


def test_copy_models(from_path, to_path, from_path_subpaths):

    # Execute the command
    copy_exp_models.copy_models(from_path, to_path)

    # No go through all from_path_subpaths and check, if they exist
    # in to_path and have the right contend
    for p in from_path_subpaths:

        # Determine the corresponding folder in to_path
        print(p.split(os.sep)[len(from_path.split(os.sep)):])
        corr_to_path = os.path.join(to_path, *p.split(os.sep)[len(from_path.split(os.sep)):])

        # Assert that the folder and the text file with the
        # right content exist
        assert os.path.isdir(corr_to_path), "Directory was not corretly made"
        assert os.path.isfile(os.path.join(corr_to_path, "test.txt")), "Directory does not contain test file"
        with open(os.path.join(corr_to_path, "test.txt")) as f:
            assert f.read() == p, "Wrong file content"
