#!/bin/bash

cd "build/"

for test in ../tests/*.yaml; do
    echo $test
    # strace -f -o "$test-trace" -I 2 ../../clustering_launcher "$test" &> ${test/".yaml"/".txt"}
    # Tests are intended to run for 5 seconds, so anything much longer indicates a hang
    timeout 15 ../../clustering_launcher "$test" &> ${test/".yaml"/".txt"}
done