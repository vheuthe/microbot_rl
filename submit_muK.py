#! /usr/bin/env python3

import numpy as np
import os
import sys



assert len(sys.argv) > 1, 'You need to specify a directory'

job_dir = sys.argv[1]

# jobnames need to start with a letter ...
job_name = 'job_' + os.path.basename(job_dir)

with open(os.path.join(job_dir, 'parameters.json'), 'r') as reader:
    job_parameters = json.load(reader)

num_tasks = np.product([len(v) for v in job_parameters.values()])

os.system('qsub -N "{}" -t 1-{} -v JOB_DIR="{}" jobscript.sh'.format(job_name, num_tasks, job_dir))
