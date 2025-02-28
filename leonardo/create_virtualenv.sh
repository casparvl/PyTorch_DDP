#!/bin/bash

module load python/3.11.6--gcc--8.5.0

python3 -m venv $HOME/pt_ddp_venv
source $HOME/pt_ddp_venv/bin/activate
python3 -m pip install torchvision
