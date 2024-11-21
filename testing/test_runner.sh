#!/bin/bash



#check if ./exec/$2 exists
if [ ! -d "./exec/$2" ]; then

    #create the directory
    mkdir -p "./exec/$2"

fi

#copy the clustering_launcher to the directory
cp ../bin/clustering_launcher "./exec/$2/clustering_launcher"

#james.yaml too
cp "../bin/james.yaml" "./exec/$2/james.yaml"

# Get absolute paths
TARGET_DIR=$(realpath "$1")
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PARENT_DIR=$(dirname "$SCRIPT_DIR")
BIN_DIR="./exec/$2"
LOGS_DIR="$SCRIPT_DIR/logs-$1"

# Check if target directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory $TARGET_DIR does not exist"
    exit 1
fi

# Check if clustering_launcher exists
if [ ! -x "$BIN_DIR/clustering_launcher" ]; then
    echo "Error: clustering_launcher not found in $BIN_DIR or not executable"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p "$LOGS_DIR"

# Process each yaml file
for yaml_file in "$TARGET_DIR"/*.yaml; do

    if [ ! -f "$yaml_file" ]; then
        echo "No YAML files found in $TARGET_DIR"
        exit 1
    fi

    filename=$(basename "$yaml_file")
    timestamp=$(date '+%Y%m%d_%H%M%S')
    dir_name=$(basename "$TARGET_DIR")
    
    echo -e "\nProcessing $filename..."
    
    # Copy yaml file to bin directory as james.yaml
    cp "$yaml_file" "$BIN_DIR/james.yaml"
    
    # Run clustering_launcher with signal handling and output redirection
    echo "Running clustering_launcher for $filename..."
    log_base="$LOGS_DIR/${dir_name}_${filename%.*}_e"
    
    # Trap and ignore SIGRTMIN (RT signal 0)
    RTMIN=$(kill -l SIGRTMIN)
    RTMAX=$(kill -l SIGRTMAX)

    # Set up traps for all RT signals
    for sig in $(seq $RTMIN $RTMAX); do
        trap '' $sig
    done

    "$BIN_DIR/clustering_launcher" "$BIN_DIR/james.yaml" \
        > "${log_base}_stdout.log" \
        2> "${log_base}_stderr.log"
    
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "Warning: clustering_launcher returned non-zero exit code: $exit_code"
    fi
    
    # Modify the third line of james.yaml
    if [ -f "$BIN_DIR/james.yaml" ]; then
        # Create a temporary file
        temp_file=$(mktemp)
        
        # Read the file line by line
        line_num=0
        while IFS= read -r line || [ -n "$line" ]; do
            line_num=$((line_num + 1))
            if [ $line_num -eq 3 ]; then
                echo "explicit_sync: false" >> "$temp_file"
            else
                echo "$line" >> "$temp_file"
            fi
        done < "$BIN_DIR/james.yaml"
        
        # Replace original file with modified content
        mv "$temp_file" "$BIN_DIR/james.yaml"

        # Run clustering_launcher with signal handling and output redirection
        echo "Running clustering_launcher for $filename..."
        log_base="$LOGS_DIR/${dir_name}_${filename%.*}_i"

        # Trap and ignore SIGRTMIN (RT signal 0)
        RTMIN=$(kill -l SIGRTMIN)
        RTMAX=$(kill -l SIGRTMAX)

        # Set up traps for all RT signals
        for sig in $(seq $RTMIN $RTMAX); do
            trap '' $sig
        done

        "$BIN_DIR/clustering_launcher" "$BIN_DIR/james.yaml" \
            > "${log_base}_stdout.log" \
            2> "${log_base}_stderr.log"

    else
        echo "Warning: james.yaml not found after running clustering_launcher"
    fi
    
    echo "Completed processing $filename"
done

echo "All processing complete"