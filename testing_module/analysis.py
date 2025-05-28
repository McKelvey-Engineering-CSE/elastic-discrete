#!/usr/bin/env python3

import os
import re
import argparse
from collections import defaultdict

def process_directory(directory):
    # Pattern to match filenames: out_tasks<number1>_run<number2>
    pattern = re.compile(r"out_tasks(\d+)_run(\d+)")
    groups = defaultdict(list)

    # Group files by the first number
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            task_id = match.group(1)
            groups[task_id].append(os.path.join(directory, filename))

    # Regular expressions to extract the needed values
    better_pattern = re.compile(
        r"Amount the constrained result worse than the optimal system state:\s*([0-9.+\-eE]+)"
    )
    worse_pattern = re.compile(
        r"Amount our result is worse than the optimal system state:\s*([0-9.+\-eE]+)"
    )

    no_answer_pattern = re.compile(
        r"Error: System is not schedulable in any configuration with specified constraints. Not updating modes."
    )

    # Process each group
    for task_id, files in sorted(groups.items(), key=lambda x: int(x[0])):
        # Stats for 'better than constrained'
        better_count = 0
        better_sum = 0.0
        constrained_no_answer = 0
        # Stats for 'worse than optimal'
        worse_count = 0
        worse_sum = 0.0
        optimal_no_answer = 0
        total_no_answer = 0

        for filepath in files:
            try:
                with open(filepath, 'r') as f:
                    for line in f:
                        # Check for "better than constrained"
                        bmatch = better_pattern.search(line)
                        if bmatch:
                            value = float(bmatch.group(1))
                            if value > 0:
                                better_count += 1
                                better_sum += value

                        # Check for "worse than optimal"
                        wmatch = worse_pattern.search(line)
                        if wmatch:
                            value = float(wmatch.group(1))
                            if value > 100:
                                optimal_no_answer += 1
                                worse_count += 1
                            elif value > 0:
                                worse_count += 1
                                worse_sum += value

                        # Check for "no answer"
                        if no_answer_pattern.search(line):
                            total_no_answer += 1

            except Exception as e:
                print(f"Warning: could not read file {filepath}: {e}")

        # Compute averages, handling division by zero
        avg_better = (better_sum / better_count) if better_count else 0.0
        avg_worse = (worse_sum / worse_count) if worse_count else 0.0

        # Output the statistics for the group
        print(f"Group {task_id}:")
        print(f"  Better than constrained: {better_count} occurrences, average = {(avg_better * 100.0):.3f}%")
        print(f"  Constrained no answer:  {constrained_no_answer} occurrences")
        print(f"  Worse than optimal:     {worse_count} occurrences, average = {(avg_worse * 100.0):.3f}%")
        print(f"  Optimal no answer:      {optimal_no_answer} occurrences")
        print(f"  Total no answer:        {total_no_answer} occurrences\n")


def main():
    parser = argparse.ArgumentParser(
        description="Process task files and compute comparison statistics."
    )
    parser.add_argument(
        "directory",
        help="Path to the directory containing out_tasks<number>_run<number> files"
    )
    args = parser.parse_args()
    process_directory(args.directory)

if __name__ == "__main__":
    main()
