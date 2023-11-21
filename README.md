# reward optimization

## installation

conda env create -f environment.yaml

python setup.py develop

## Running DDPO
accelerate launch --num_processes=2  scripts/ddpo.py --config config/ddpo_config.py:aesthetic &

## Running AlignProp

## Gathering dataset for offline algorithms
accelerate launch --num_processes=2 scripts/offline_sampling.py --num_samples 25600 --batch_size_per_device 32 &

## Running RWR
accelerate launch --num_processes=2  scripts/wrl.py --config config/wrl_config.py:aesthetic &

accelerate launch --num_processes=2  scripts/online_wrl.py --config config/owrl_config.py:aesthetic 

## Running RCG




# enviroment setup

conda create -n ropt11 python==3.11.5
conda activate ropt11
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install jupyterlab
pip install ml-collections absl-py diffusers[torch] transformers wandb inflect pydantic ipdb matplotlib accelerate