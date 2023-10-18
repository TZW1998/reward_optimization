# reward optimization

## installation

conda env create -f environment.yaml

python setup.py develop

## Running DDPO
accelerate launch --num_processes=2  scripts/ddpo.py --config config/ddpo.py:aesthetic &

## Running AlignProp

## Gathering dataset for offline algorithms
accelerate launch --num_processes=2 scripts/offline_sampling.py --num_samples 25600 --batch_size_per_device 32 &

## Running RWR
accelerate launch --num_processes=2  scripts/wrl.py --config config/wrl.py:aesthetic &

## Running RCG