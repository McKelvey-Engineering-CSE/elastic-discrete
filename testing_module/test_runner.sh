#!/bin/bash

# Ignore SIGUSR1 and SIGTERM signals
trap '' SIGUSR1 SIGTERM

# Set the static first parameter
NUMBER_1=100

# Loop from 6 to 20 in increments of 2
for NUMBER_2 in $(seq 5 1 25); do
    echo "Running: python3 ./test_runner.py tasks $NUMBER_1 $NUMBER_2"
    python3 ./test_runner.py tasks $NUMBER_1 $NUMBER_2
done