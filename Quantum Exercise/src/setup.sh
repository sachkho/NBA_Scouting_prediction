#!/bin/bash

# setup.sh

# Check if the virtual environment exists, create it if not
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

# Activate the virtual environment
source .venv/bin/activate

# Install pipreqs if not already installed
pip install pipreqs

# Install the required packages
pip install -r src/requirements.txt

# Install pygraphviz
pip install pygraphviz --install-option="--include-path=/usr/local/include/graphviz" \
                       --install-option="--library-path=/usr/local/lib/graphviz"

# Run the database creation script
python src/database.py
