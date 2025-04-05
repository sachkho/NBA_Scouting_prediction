# setup.ps1

# Check if the virtual environment exists, create it if not
if (-Not (Test-Path -Path ".\.venv")) {
    python -m venv .\.venv
}

# Activate the virtual environment
& .\.venv\Scripts\Activate.ps1

# Install pipreqs if not already installed
pip install pipreqs

# Install the required packages
pip install -r .\src\requirements.txt

# Install pygraphviz
python -m pip install --config-settings="--global-option=build_ext" `
              --config-settings="--global-option=-IC:\Program Files\Graphviz\include" `
              --config-settings="--global-option=-LC:\Program Files\Graphviz\lib" `
              pygraphviz

# Run the database creation script
python .\src\database.py