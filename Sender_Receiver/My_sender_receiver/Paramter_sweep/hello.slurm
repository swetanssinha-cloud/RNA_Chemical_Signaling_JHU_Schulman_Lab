#!/bin/bash
#SBATCH --job-name=hello
#SBATCH --output=hello.%j.out
#SBATCH --time=00:02:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=express

echo "Hello from $(hostname)!"
sleep 30