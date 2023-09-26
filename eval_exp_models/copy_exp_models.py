'''
This script copies all the experimentally
trained models found in a specific folder
to the same file structure in another spe-
cific folder
'''
import os
import shutil
from pathlib import Path

def copy_models(from_path, to_path):
    '''
    Function for copying all models found in
    from_folder to to_folder in the same dir
    structure they are found in
    '''

    # Find all models in the from_path
    model_paths = find_models(from_path)

    # Make a path object from from_path and to_path
    source_path = Path(from_path)
    dest_path = Path(to_path)

    # Now copy the directories that contain
    # models to a corresponding directory in
    # to_path replicating the dir tree under
    # from_path
    for model_path in model_paths:

        # Make a path object
        mod_path = Path(model_path)

        # Get the path after from_path (to
        # recreate that structure in the
        # to_path)
        sub_path = os.path.join(*mod_path.parts[len(source_path.parts):])

        # Check, if the path already exists
        # and if so do not attempt to copy
        # that one
        if not (dest_path/sub_path).is_dir():
            shutil.copytree(model_path, (dest_path/sub_path))
            print(f'Copied {model_path} to {dest_path/sub_path}')
        else:
            print(f'Path {sub_path} already exists in {to_path}')

        # Don't forget the parameters
        if not ((dest_path/sub_path).parent/'parameters.json').is_file():
            shutil.copy((mod_path.parent/'parameters.json'), (dest_path/sub_path).parent)
            print(f'Copied parameters from {Path(model_path).parent} to {(dest_path/sub_path).parent}')
        else:
            print(f'Parameters for {sub_path} already exist in {to_path}')



def find_models(path):
    '''
    Function for finding all models in a spec-
    ific path, returns absolute paths to all
    models
    '''

    # Search for folders being named "model_actor"
    # or "model_critic"
    model_paths = []
    for root_dir, dirs, files in os.walk(path):

        # The path is a valid model, if it has
        # either "model_actor" or "model_critic"
        # in its name and contains "saved_model.pb"
        if "model_" in root_dir \
            and (any("saved_model.pb" in f for f in files) \
                or any("test.txt" in f for f in files)):
            model_paths.append(root_dir)

            # Make dirs empty, so os.walk does not
            # descend any further
            dirs[:] = []

    return model_paths


if __name__ == "__main__":
    scc_path = "/data/scc/veit-lorenz.heuthe/rod_project/exp_trained_models/2023/04"
    lab_path = "/mnt/share/interactionsetup/Data/2023/04"
    copy_models(from_path=lab_path, to_path=scc_path)