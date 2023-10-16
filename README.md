# reward optimization

## installation

conda env create -f environment.yaml

python setup.py develop

## Running DDPO
accelerate launch --num_processes=2  scripts/ddpo.py --config config/ddpo.py:aesthetic &

## Running AlignProp

## Gathering dataset for offline algorithms

## Running RWR

## Running RCG