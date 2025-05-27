#!/usr/bin/env python3
import argparse
import glob
import os
import random
import re
import subprocess
import sys
import signal

#task_set_skipped
skipped_set = False

def skip_set(who, cares):

    global skipped_set

    skipped_set = True

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='YAML task selector and launcher')
    parser.add_argument('folder', help='Folder containing YAML files')
    parser.add_argument('num_runs', type=int, help='Number of runs to execute')
    parser.add_argument('tasks_to_select', type=int, help='Number of tasks to select')
    return parser.parse_args()

def count_tasks_in_file(file_path):
    """Count the number of task entries in a YAML file."""
    task_count = 0
    with open(file_path, 'r') as f:
        content = f.read()
        # Count entries that start with "  - program:"
        task_count = len(re.findall(r"^\s+-\s+program:", content, re.MULTILINE))
    return task_count

def extract_task_at_index(file_path, task_index):
    """Extract a specific task from a file based on its index."""
    task_text = ""
    current_task = -1
    in_task = False
    task_indent = 0
    
    with open(file_path, 'r') as f:
        for line in f:
            # Detect start of a task
            if re.match(r"^\s+-\s+program:", line):
                current_task += 1
                if current_task == task_index:
                    in_task = True
                    task_text = line
                    task_indent = len(line) - len(line.lstrip())
                elif current_task > task_index and in_task:
                    # We've moved past our target task
                    break
            elif in_task:
                # Check if we're still within the same task by indent level
                if line.strip() and len(line) - len(line.lstrip()) <= task_indent:
                    if not line.startswith(' ' * task_indent + '- '):
                        break
                task_text += line
    
    # Parse the task text to a dictionary structure
    task_dict = parse_task_yaml(task_text)
    return task_dict

def parse_task_yaml(task_text):
    """
    Parse a YAML task text to a dictionary structure manually.
    This is a simplified parser for the specific YAML format.
    """
    # Extract key properties from the text
    program_name_match = re.search(r"name:\s+(\w+)", task_text)
    program_args_match = re.search(r'args:\s+"([^"]+)"', task_text)
    elasticity_match = re.search(r"elasticity:\s+(\d+)", task_text)
    
    # Initialize the task dictionary
    task = {
        "program": {
            "name": program_name_match.group(1) if program_name_match else "unknown",
            "args": program_args_match.group(1) if program_args_match else "0"
        },
        "elasticity": int(elasticity_match.group(1)) if elasticity_match else 2,
        "modes": []
    }
    
    # Extract modes
    mode_blocks = re.findall(r"- work:.*?(?=- work:|$)", task_text, re.DOTALL)
    for block in mode_blocks:
        if not block.strip():
            continue
            
        mode = {}
        
        # Extract each time component with regex
        fields = ["work", "span", "gpu_work", "gpu_span", "period"]
        for field in fields:
            sec_match = re.search(fr"{field}:.*?sec:\s+(\d+)", block)
            nsec_match = re.search(fr"{field}:.*?nsec:\s+(\d+)", block)
            
            if sec_match and nsec_match:
                mode[field] = {
                    "sec": int(sec_match.group(1)),
                    "nsec": int(nsec_match.group(1))
                }
        
        if mode:  # Only add non-empty modes
            task["modes"].append(mode)
    
    return task

def get_total_tasks(folder_path):
    """Get the total number of tasks and their distribution in all YAML files."""
    yaml_files = glob.glob(os.path.join(folder_path, "*.yaml"))
    
    files_info = []
    total_tasks = 0
    
    for file_path in yaml_files:
        task_count = count_tasks_in_file(file_path)
        files_info.append({"path": file_path, "tasks": task_count, "start": total_tasks})
        total_tasks += task_count
    
    return total_tasks, files_info

def select_random_tasks(total_tasks, files_info, count):
    """Randomly select tasks without loading them all."""
    selected_indices = random.sample(range(total_tasks), count)
    selected_indices.sort()  # Sort for more efficient file reading
    
    selected_tasks = []
    for index in selected_indices:
        # Find which file this index belongs to
        file_info = next((f for f in files_info if f["start"] <= index < f["start"] + f["tasks"]), None)
        if file_info:
            file_index = index - file_info["start"]
            task = extract_task_at_index(file_info["path"], file_index)
            selected_tasks.append(task)
    
    return selected_tasks

def create_james_yaml(selected_tasks, task_count, bin_dir="bin", output_file="james.yaml"):
    """Create james.yaml with selected tasks in the bin directory."""
    # Replace the name and args in selected tasks
    for task in selected_tasks:
        if 'program' in task:
            task['program']['name'] = 'james'
            task['program']['args'] = '0 61428571 2 10'
    
    # Create the YAML content as text
    yaml_content = "--- \nschedulable: true\nexplicit_sync: false\nmaxRuntime: {sec: 60, nsec: 0}\ntasks:\n"
    
    # Add each task
    for task in selected_tasks:
        yaml_content += "  - program:\n"
        yaml_content += f"      name: {task['program']['name']}\n"
        yaml_content += f"      args: \"{task['program']['args']}\"\n"
        yaml_content += f"    elasticity: {task['elasticity']}\n"
        yaml_content += "    modes:\n"
        
        # Add each mode
        for mode in task['modes']:
            yaml_content += "      - "
            
            # Add each field in the mode
            first_field = True
            for field_name, value in mode.items():
                if first_field:
                    yaml_content += f"{field_name}: {{sec: {value['sec']}, nsec: {value['nsec']}}}"
                    first_field = False
                else:
                    yaml_content += f"\n        {field_name}: {{sec: {value['sec']}, nsec: {value['nsec']}}}"
            yaml_content += "\n"
    
    # Ensure bin directory exists
    os.makedirs(bin_dir, exist_ok=True)
    
    # Full path to output file in bin directory
    output_path = os.path.join(bin_dir, output_file)
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write(yaml_content)
    
    return output_path

def run_clustering_launcher(yaml_file, task_count, run_number, bin_dir="bin"):
    """Run the clustering launcher command from the bin directory."""
    # Extract just the filename without the path
    yaml_filename = os.path.basename(yaml_file)
    
    # Format the output filename based on task count and run number
    output_file = f"out_tasks{task_count}_run{run_number}"
    
    # Construct the command to run from the bin directory
    command = f"cd {bin_dir} && ./clustering_launcher ./{yaml_filename} SIM &> ../{output_file}"
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error executing clustering launcher: {e}")
        return 1

def main():
    args = parse_args()

    global skipped_set
    
    # Setup consistent seed once at the beginning
    seed = args.num_runs * args.tasks_to_select
    random.seed(seed)
    print(f"Using seed: {seed}")
    
    # Get information about tasks in all files
    total_tasks, files_info = get_total_tasks(args.folder)
    print(f"Found {total_tasks} tasks across all files")
    
    if total_tasks < args.tasks_to_select:
        print(f"Error: Not enough tasks to select {args.tasks_to_select} tasks")
        sys.exit(1)
    
    # Define bin directory
    bin_dir = "bin"
    if not os.path.isdir(bin_dir):
        print(f"Warning: '{bin_dir}' directory not found. Creating it.")
        os.makedirs(bin_dir, exist_ok=True)

    # Set python to ignore SIGUSR1 since that is used by the scheduler
    signal.signal(signal.SIGUSR1, skip_set)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    
    # Run the specified number of times
    for run in range(1, args.num_runs + 1):
        print(f"\nStarting run {run}/{args.num_runs}")

        return_code = 2

        # Loop until we get a valid set
        while return_code == 2:
        
            # Randomly select tasks - no need to reseed, Python's random maintains state
            selected_tasks = select_random_tasks(total_tasks, files_info, args.tasks_to_select)
            
            # Create james.yaml inside bin directory
            yaml_file = create_james_yaml(selected_tasks, args.tasks_to_select, bin_dir)
            print(f"Created {yaml_file} with {len(selected_tasks)} tasks")
            
            # Run clustering launcher
            return_code = run_clustering_launcher(yaml_file, args.tasks_to_select, run, bin_dir)

            # If return code is 2, task set is not scheduable at all
            if skipped_set == True:
                print(f"Error: clustering_launcher returned unscheduable. Generating new taskset.")
                return_code = 2
                skipped_set = False

            # If something went wrong
            elif return_code != 0:
                print(f"Error: clustering_launcher returned {return_code}. Stopping execution.")
                sys.exit(1)
        
        print(f"Run {run} completed successfully")
    
    print(f"\nAll {args.num_runs} runs completed successfully")

if __name__ == "__main__":
    main()