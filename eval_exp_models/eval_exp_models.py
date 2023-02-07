'''
This copies all exprimental models to the scc
and evalutes them there by running a bunch of evaluation
episodes
'''
import json
import os
from pathlib import Path
import copy_exp_models


def eval_exp_models(models_path, eval_path):
    '''
    This copies all exprimental models from models_path
    to eval_path and evalutes them there by running a
    bunch of evaluation episodes
    '''

    # Copy all experimental models to the evaluation file tree
    # copy_exp_models.copy_models(from_path=models_path, to_path=eval_path)

    # Now iterate through all models and evaluate them
    all_models = Path(eval_path).glob(pattern="**/parameters.json")
    for mod_p in all_models:

        # Make sure that model is not evaluated, yet
        if (mod_p.parent/'evaluation.h5').is_file():
            print(f'Model {mod_p} is already evaluated')
            continue

        # Modify the parameters so there is only evaluation
        # and no training (exclude model_structure and host adress,
        # since they are always lists)
        with (mod_p).open(mode='r') as paramfile:
            parameters = json.load(paramfile)
        parameters.update({'train_ep': 0, 'eval_ep': 8, 'load_models': str(mod_p.parent/'model')})

        # Pop out the model structure and the host adress,
        # since they make trouble as they are lists
        parameters.pop('model_structure')
        parameters.pop('host_address')

        # Save the modified parameters
        with (mod_p).open(mode='w') as paramfile:
            json.dump(parameters, paramfile, ensure_ascii=False, indent=4)

        # Now run call submit_dir to evaluate the model
        command = f"python submit_dir.py {str(mod_p.parent)}"
        os.system(command)


if __name__ == "__main__":
    from_path = "/mnt/share/interactionsetup/Data/2023/01"
    to_path = "/data/scc/veit-lorenz.heuthe/rod_project/exp_trained_models/2023/01"
    eval_exp_models(models_path=from_path, eval_path=to_path)
