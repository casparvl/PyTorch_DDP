#!/bin/bash

module load python/3.11.6--gcc--8.5.0

python3 -m venv pt_ddp_venv
source pt_ddp_venv/bin/activate
python3 -m pip install torchvision
