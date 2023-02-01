import os
import shutil

# This script copies all the experimentally
# trained models found in a specific folder
# to the same file structure in another spe-
# cific folder

def copy_models(from_path, to_path):
    '''
    Function for copying all models found in
    from_folder to to_folder in the same dir
    structure they are found in
    '''

    # Find all models in the from_path
    model_paths = find_models(from_path)

    # Now copy the directories that contain
    # models to a corresponding directory in
    # to_path replicating the dir tree under
    # from_path
    for mp in model_paths:

        # Get the path after from_path (to
        # recreate that structure in the
        # to_path)
        sub_path = os.path.join(*mp.split(os.sep)[len(from_path.split(os.sep)):])

        # Check, if the path already exists
        # and if so do not attempt to copy
        # that one
        if os.path.isdir(os.path.join(to_path, sub_path)):
            print(f'Path {sub_path} already exists in {to_path}')
            continue

        # If not, copy
        shutil.copytree(mp, os.path.join(to_path, sub_path))
        print(f'Copied {mp} to {os.path.join(to_path, sub_path)}')


def find_models(path):
    '''
    Function for finding all models in a spec-
    ific path, returns absolute paths to all
    models
    '''

    # Search for folders being named "model_actor"
    # or "model_critic"
    model_paths = []
    for root_dir, _, files in os.walk(path):

        # The path is a valid model, if it has
        # either "model_actor" or "model_critic"
        # in its name and contains "saved_model.pb"
        if "model_" in root_dir \
            and (any(["saved_model.pb" in f for f in files]) \
                or any(["test.txt" in f for f in files])):
            model_paths.append(root_dir)

    return model_paths

if __name__ == "__main__":
    from_path = os.path.abspath("./from_folder_test")
    to_path = os.path.abspath("./to_folder_test")
    model_paths = find_models(from_path)
    copy_models(from_path=from_path, to_path=to_path)
