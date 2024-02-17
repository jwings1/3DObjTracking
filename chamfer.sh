#!/bin/bash

#SBATCH --mail-type=ALL                           # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --job-name="ROMP_ev"
#SBATCH --error=/scratch_net/biwidl307/lgermano/3DTrackEval/log/error/%j.err
#SBATCH --output=/scratch_net/biwidl307/lgermano/3DTrackEval/log/out/%j.out
#SBATCH --mem=30G
#SBATCH --gres=gpu:5

# Set a directory for temporary files unique to the job with automatic removal at job termination
TMPDIR=$(mktemp -d)
if [[ ! -d ${TMPDIR} ]]; then
    echo 'Failed to create temp directory' >&2
    exit 1
fi
trap "exit 1" HUP INT TERM
trap 'rm -rf "${TMPDIR}"' EXIT
export TMPDIR

# Change the current directory to the location where you want to store temporary files, exit if changing didn't succeed.
# Adapt this to your personal preference
cd "${TMPDIR}" || exit 1

# Send some noteworthy information to the output log
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

source /itet-stor/lgermano/net_scratch/conda/etc/profile.d/conda.sh
conda activate chamfer
cd /scratch_net/biwidl307/lgermano/3DTrackEval
export HOME=/scratch_net/biwidl307_second/lgermano
CONDA_OVERRIDE_CUDA=11.8

# cd pytorch3d
# pip install -e .

cd /scratch_net/biwidl307/lgermano/3DTrackEval
python chamfer_evaluate.py

echo "DONE!"

# Send more noteworthy information to the output log
echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0
