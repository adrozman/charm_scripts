#!/bin/bash -l

#$ -P turbomac    		# project to charge hours to
#$ -l h_rt=6:00:00		# how long you are requesting (HH:MM:SS)
#$ -N joby_directivity  # job name that will be displayed by qstat or in email
#$ -j y				    # combine error and log file into one file
#$ -pe omp 16           # use n cores
#$ -m abe

# first, cd to the directory where you qsubbed from.
cd $SGE_O_WORKDIR

# you will need to source scc_vars to add charm executables to path. You can change this to a different file in your own directory if you need to modify it. 
source /project/turbomac/CHARM/scc_vars.sh

# maybe you want to clear out
#rm -r charm_runs

# sets up CHARM cases, runs them, adjusts the WOPWOP inputs, runs WOPWOP
python charm_joby_runs.py

# parses through the run folders, extracts pressures, Fourier transforms, saves directivity plots
# you can re-run this script from terminal if you want to remake the plots without running charm again
#python charm_BPF_plots.py
