# =============================================================================
# Script Name: database.py
# =============================================================================

"""
Purpose:
--------------------------------------------------------------------------------
This module establishes the database connection, creates tables if they do not
exist, and provides session management for the project. The tables are created
based on the ORM models defined in models.py and populated from CSV files in
the data/tbls folder if the database is being initialized.

Structure:
--------------------------------------------------------------------------------
1. Database Engine and Session Management - Functions to create the database
    connection, initialize the database if necessary, and manage database 
    sessions.


Usage:
--------------------------------------------------------------------------------
This module is automatically executed when the project is initialized. It sets
up the database connection and ensures the database is correctly structured and
populated based on the data in the data/tbls folder.

Dependencies:
--------------------------------------------------------------------------------
Ensure all dependencies listed in requirements.txt are installed. This module
requires models.py for ORM class definitions.

Author:
--------------------------------------------------------------------------------
Roberto Primo Curti
[2024-07-25]
"""

#################################### Imports ####################################

import os
import pandas as pd
from config import ROOT_FOLDER, LIST_PATHS
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
import models as mdl

################################## Environment ##################################
# Variables
tbl_folder = LIST_PATHS['data\\tbls']
database_path = os.path.join(ROOT_FOLDER, 'basketballDraftProject.db')

# Create the engine
engine = sa.create_engine(f'sqlite:///{database_path}')

# Create tables if they don't exist
if not os.path.exists(database_path):
    # Load CSV files into the database
    tables = {}

    # Store the files in the data/tbls folder in a dictionary
    for root, dirs, files in os.walk(tbl_folder):
        for file in files:
            if file.endswith('.csv'):
                # Add the file to the dictionary as {filename}:{filepath}
                tables[file.split('.')[0]] = os.path.join(root, file)

    # Load CSV files into the database
    for table_name, csv_file in tables.items():
        df = pd.read_csv(csv_file)
        # Rename columns if needed
        df.columns = df.columns.str.replace('.', '_')
        df.to_sql(table_name, con=engine, if_exists='replace', index=False)

    # Create all tables in the database
    mdl.Base.metadata.create_all(engine)

# Create a configured "Session" class
Session = sessionmaker(bind=engine)
# Create a Session
session = Session()

