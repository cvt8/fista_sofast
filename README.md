# Reproduction of the code of the paper "FISTA algorithm: so fast ?" by G. Fort, L. Risser, Y. Atchadé and E. Moulines
Authors: Mathis Scheffler and Constantin Vaillant-Tenzer

## How to Run the Code

To generate the video from our code, follow these steps:

1. Clone the project repository: `$ git clone https://github.com/cvt8/fista_sofast.git`
2. Navigate to the repository: `$ cd fista_sofast` (for Linux users)
3. Create a new Python environment: `$ conda env create -f environment.yml`
4. Activate the conda environment: `$ conda activate fista`
5. Run the code: `$ python fista_use_paper.py`
5. (bis) alternatively you can use slurm: `$ sbatch laucha_fista.sh`

## Contents

`presentation.pdf`: a beamer presentation of our work

`fisa_use_paper.py`: main code to reproduce the results of our paper

`grad.py`: helper function for sampling (Wolff sampling), gradient and penalty computation

`visualisation.py`: Graph plotting functions

`base_fista.py`: a basic implementation of the fista algorithm and others algorithm (we finally don't use this code)

`logs`: logs when implementing our code

`lauch_fista.sh`: shell script if you want to run our code n a computation cluster using slurm (to adapt with the cluster parameters)

## Possible improvements

As the code take hours (even days) to run, using C instead of python would dramatically increase the compting speed (about 87 times for the pi-decimals benchmark: https://github.com/niklas-heer/speed-comparison).