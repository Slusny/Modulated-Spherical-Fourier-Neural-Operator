#!/bin/bash

# This script removes the files from the cluster recursively from all subdirectories of the given directory $1
shopt -s globstar 
for f in $(find $1 -type f); do
    if [ -f $f ]; then
        #rm $f
        echo "Removing file $f"
        : > $f
        rm -f $f
    fi
done
