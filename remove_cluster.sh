#!/bin/bash

# This script removes the files from the cluster recursively from all subdirectories of the given directory $1
shopt -s globstar 
for f in $(ls -d $1/** ); do
    if [ -f $f ]; then
        #rm $f
        echo "Removing file $f"
    fi
done