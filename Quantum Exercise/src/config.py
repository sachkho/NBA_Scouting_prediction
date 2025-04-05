# =============================================================================
# Script Name: config.py
# =============================================================================

"""
Purpose:
--------------------------------------------------------------------------------
This module defines constants and contains functions specifically required for 
initializing and managing these settings. It serves as a central configuration 
file to set project-wide parameters, paths, and other configuration settings 
that are critical for the project's Python modules to function consistently and 
correctly.

Structure:
--------------------------------------------------------------------------------
1. Constant Definitions - Defining constants that will be used across multiple 
   modules within the project.
2. Configuration Functions - Functions designed to initialize or modify the 
   configuration settings, including setting paths dynamically, reading 
   configuration from external files, or environment variables.
3. System functions - Functions designed to support the code excecution, but 
   that are not directly related to statistical, analytical tasks, or 
   related to the research itself.


Usage:
--------------------------------------------------------------------------------
This module should be imported at the beginning of each Python module in the 
project that requires access to the configuration settings defined here. It 
ensures a unified configuration across all scripts and facilitates maintenance 
and updates to project settings.

Example:
from config import SOME_CONSTANT, someFunction

Dependencies:
--------------------------------------------------------------------------------
Ensure all dependencies listed in `requirements.txt` are installed.

Author:
--------------------------------------------------------------------------------
Roberto Primo Curti
[2024-07-24]
"""

#################################### Imports ####################################

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

################################### Functions ###################################

def find_project_root():
    """
    Searches for the root directory of the current project by looking for a `.gitignore` file.
    It starts from the current working directory and moves up through parent directories
    until it finds an `.gitignore` file. If such a file is found, the directory containing
    the file is considered the root directory of the project. If no `.gitignore` file is found,
    returns None.
    
    Returns:
        str or None: The absolute path to the project's root directory if an .gitignore file 
        is found, otherwise None.
    """
    current_dir = Path(os.getcwd())
    while current_dir != current_dir.parent:
        if any(current_dir.glob("*.gitignore")):
            return str(current_dir)
        current_dir = current_dir.parent
    return None

def find_subdirs(path, path_rel="", exc_dirs=None):
    """
    Searches for all subdirectories within a given directory, excluding specified directories.
    It can recursively explore each directory and subdirectory, building a path list
    excluding those specified in `exc_dirs`.
    
    Args:
        path (str): The base directory from which to start the search.
        path_rel (str): The relative path to use as the prefix in the resulting list of
                        directories. Default is an empty string, indicating no prefix.
        exc_dirs (list of str): List of directory names to exclude from the search.
                                Defaults to [".venv", ".git", "__pycache__"].
    
    Returns:
        dict: A dictionary where each key is the relative path (from the base
              directory specified by `path` and `path_rel`) to a subdirectory, and
              each value is the absolute path to that subdirectory. Directories
              listed in `exc_dirs` are excluded.
    """
    if exc_dirs is None:
        exc_dirs = [".venv", ".git", "__pycache__"]
    base_path = Path(path)
    result = {}
    
    for dirpath, dirnames, _ in os.walk(base_path):
        dirnames[:] = [d for d in dirnames if d not in exc_dirs]  
        for dirname in dirnames:
            full_path = Path(dirpath) / dirname
            rel_path = os.path.relpath(full_path, base_path)
            result[rel_path] = str(full_path)
            
    return result

def list_subdirectories(path_root, exc_dirs=None):
    """
    Initiates a recursive search for subdirectories from a specified root directory,
    excluding certain directories as specified.
    
    Args:
        path_root (str): The root directory from which the recursive search should start.
        exc_dirs (list of str): List of directory names to exclude from the search.
                                Defaults to ["env"].
    
    Returns:
        dict: A named list of subdirectories found, excluding those specified in `exc_dirs`.
              The list's names are relative paths from `path_root`, and its values are the
              absolute paths to those subdirectories.
    """
    return find_subdirs(path_root, exc_dirs=exc_dirs)


################################### Global Variables ###################################
ROOT_FOLDER = find_project_root()
if ROOT_FOLDER is None:
    raise FileNotFoundError("Project root directory could not be found. Ensure you have a .gitignore file in your project root.")

LIST_PATHS = find_subdirs(ROOT_FOLDER)

# Debugging: Print the keys and values
# print("Content of LIST_PATHS:")
# for key, value in LIST_PATHS.items():
#     print(f"{key}: {value}")

# # Checking specific keys
# ext_data = LIST_PATHS.get('data\\ext', 'Key not found')
# table_data = LIST_PATHS.get('data\\tables', 'Key not found')

# print(f'ext_data: {ext_data}')
# print(f'table_data: {table_data}')

DATA_FOLDER = LIST_PATHS["data"]
RAW_DATA_FOLDER = LIST_PATHS["data\\raw"]
IMG_FOLDER = LIST_PATHS["img"]
SRC_FOLDER = LIST_PATHS["src"]
