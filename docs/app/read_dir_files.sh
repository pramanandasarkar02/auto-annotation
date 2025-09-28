#!/bin/bash
# Concatenate all files in a directory

DIR=${1:-.}   # default is current directory

for file in "$DIR"/*; do
    if [[ -f "$file" ]]; then
        echo "===== $file ====="
        cat "$file"
        echo -e "\n"
    fi
    if 

done
