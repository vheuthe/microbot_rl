#! /usr/bin/env python3

import json
import numpy as np
import os
import sys


assert len(sys.argv) > 1, 'You need to specify a directory'

job_dir = sys.argv[1]

# jobnames need to start with a letter ...
job_name = 'job_' + os.path.basename(job_dir)

with open(os.path.join(job_dir, 'parameters.json'), 'r') as reader:
    job_parameters = json.load(reader)

# warning: types like strings also have a len()!
num_tasks = int(np.product([len(v) for v in job_parameters.values() if isinstance(v, list)]))

os.system(f'qsub -N "{job_name}" -t 1-{num_tasks} -v JOB_DIR="{job_dir}" jobscript.sh')
