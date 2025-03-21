#!/bin/bash
conda -V
#conda init 
source /home/idies/miniconda3/envs/py39/etc/profile.d/conda.sh
conda activate /home/idies/workspace/Storage/cdavies/persistent/envs
conda deactivate
conda activate /home/idies/workspace/Storage/cdavies/persistent/envs
python -m venv .venv
#source .venv/bin/activate 
pip install -r /home/idies/workspace/Storage/cdavies/persistent/requirements.txt
python -m ipykernel install --user --name py312  --display-name "py312"
huggingface-cli login --token $HF_TOKEN --add-to-git-credential

