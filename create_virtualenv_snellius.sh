#!/bin/bash

module load 2024
module load Python/3.12.3-GCCcore-13.3.0

python3 -m venv pt_ddp_venv
source pt_ddp_venv/bin/activate
python3 -m pip install torchvision
