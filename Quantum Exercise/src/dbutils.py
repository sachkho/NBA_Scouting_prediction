# =============================================================================
# Script Name: database.py
# =============================================================================

"""
Purpose:
--------------------------------------------------------------------------------
This module established the script to create the database and tables for the
project from the csv files in the data/tbls folder. It also contains functions
to interact with the database, such as inserting data into the tables, querying
data from the tables, and deleting data from the tables.

Structure:
--------------------------------------------------------------------------------
1. Creating the database - Functions to create the database and tables from the
    csv files in the data/tbls folder.


Usage:
--------------------------------------------------------------------------------
This module is automatically executed when the project is initialized. It creates
the database and tables required for the project. The database and tables are
created from the csv files in the data/tbls folder. The model is also able to be
sourced at the start of other scripts and notebooks to make use of the database
interactions.

Example:
from database import SOME_CONSTANT, someFunction

Dependencies:
--------------------------------------------------------------------------------
Ensure all dependencies listed in `requirements.txt` are installed.

Author:
--------------------------------------------------------------------------------
Roberto Primo Curti
[2024-07-25]
"""

#################################### Imports ####################################
import os
import pandas as pd
import sqlalchemy as sa
from database import engine

################################## Environment ##################################
# Variables

################################### Functions ###################################

def view_playerStats(list_id=None, drop_id=True):
    """
    Returns a dataframe containing the player statistics for all players in the
    database. The function is used to create a view of statistics for all players
    in the database while replacing IDs with their respective labels. If a list
    of player IDs is provided, the function will only return the statistics for
    those players.

    Args:
        list_id (list of int): A list of player IDs. Default is None.
        drop_id (bool): A boolean to drop the ID columns.

    Returns:
        pd.DataFrame: A dataframe containing the player statistics.
    """

    # Define the base query to select the player statistics
    query = """
    SELECT
        ph.season_year AS year,
        ph.season_stage AS stage,
        t.team_id AS team_id,
        t.team_name AS team,
        p.player_id AS player_id,
        p.player_name AS player,
        SUM(ph.stat_games) AS games,
        SUM(ph.stat_gamesStart) AS gamesStarted,
        SUM(ph.stat_minutes) AS minutes,
        SUM(ph.stat_fg) AS fg,
        SUM(ph.stat_fga) AS fga,
        AVG(ph.stat_fgPct) AS fgPct,
        SUM(ph.stat_fg3) AS fg3,
        SUM(ph.stat_fg3a) AS fg3a,
        AVG(ph.stat_fg3Pct) AS fg3Pct,
        SUM(ph.stat_fg2) AS fg2,
        SUM(ph.stat_fg2a) AS fg2a,
        AVG(ph.stat_fg2Pct) AS fg2Pct,
        AVG(ph.stat_efgPct) AS efgPct,
        SUM(ph.stat_ft) AS ft,
        SUM(ph.stat_fta) AS fta,
        AVG(ph.stat_ftPct) AS ftPct,
        SUM(ph.stat_orb) AS orb,
        SUM(ph.stat_drb) AS drb,
        SUM(ph.stat_trb) AS trb,
        SUM(ph.stat_ast) AS ast,
        SUM(ph.stat_stl) AS stl,
        SUM(ph.stat_blk) AS blk,
        SUM(ph.stat_tov) AS tov,
        SUM(ph.stat_pf) AS pf,
        SUM(ph.stat_pts) AS pts
    FROM
        playerHistory ph
    JOIN
        teams t ON ph.team_id = t.team_id
    JOIN
        players p ON ph.player_id = p.player_id
    """
    
    # Add WHERE clause if list_id is provided
    if list_id:
        placeholders = ', '.join(['?'] * len(list_id))
        query += f" WHERE p.player_id IN ({placeholders})"
    
    # Add the GROUP BY clause
    query += """
    GROUP BY 
        ph.season_year, ph.season_stage, t.team_id, p.player_id, t.team_name, p.player_name
    """

    # Execute the query
    data = pd.read_sql(query, engine, params=tuple(list_id) if list_id else None)

    # Create a view if no list_id is provided
    if not list_id:
        data.to_sql('view_playerStats', con=engine, if_exists='replace', index=False)

    # Order by year then by team then by player
    data.sort_values(by=['year', 'team', 'player'], inplace=True)

    # Drop unnecessary columns
    if drop_id:
        data.drop(columns=['player_id', 'team_id'], inplace=True)

    return data

def get_playerID(df):
    """
    Returns a list of player IDs from a dataframe consisting of player name, team
    name, and year. The function is used to get the player ID for a given player
    assisted by the combination of team name and year, to ensure the correct
    player.
    
    Args:
        df['player'] (str): The name of the player. This must be in lowercase 
                                    and utf-8 encoded.
        df['team'] (str): The name of the team. This must be in lowercase and 
                                    utf-8 encoded.
        df['year'] (int): The year the player played for the related team.
    
    Returns:
        list: A list of player IDs.
    """

    # Create an inspector object
    inspector = sa.inspect(engine)
    
    # Check if playerStats view exists
    if 'playerStats' not in inspector.get_table_names():
        view_playerStats()
    
    # Stpre playerStats
    data = pd.read_sql('SELECT * FROM playerStats', engine)

    # Keep only year, team_id, team, player_id,  player columns
    data = data[['year', 'team_id', 'team', 'player_id', 'player']]

    # Merge the dataframes on year, team [df is team ] and player
    data = pd.merge(df, data, how='left', on=['year', 'team', 'player'])

    # Print number of rows with missing player_id
    # print(f"Missing player_id: {data['player_id'].isnull().sum()}")

    # Store only player_id
    data = data['player_id']

    return data.tolist()


def view_draftHistory(drop_id=False):
    """
    Returns a dataframe containing the draft results for all years in the
    database. The function is used to create a view of draft results
    in the database while replacing IDs with their respective labels.

    Args:        
        drop_id (bool): A boolean to drop the ID columns.

    Returns:
        pd.DataFrame: A dataframe containing the draft results.
    """

    # Define the base query to select the draft results
    query = """
            SELECT
                dh.season_year AS year,
                t.team_id AS team_id,
                t.team_name AS team,
                p.player_id AS player_id,
                p.player_name AS player,
                dh.draft_round AS round,
                dh.draft_pickRound AS pick_round,
                dh.draft_pickOverall AS pick_overall
            FROM 
                draftHistory dh
            LEFT JOIN 
                teams t ON dh.team_id = t.team_id
            LEFT JOIN 
                players p ON dh.player_id = p.player_id            
            """
    
    # Execute the query
    data = pd.read_sql(query, engine)    

    # Order by year then by team then by player
    data.sort_values(by=['year', 'pick_overall'], inplace=True)

    # Drop unnecessary columns
    if drop_id:
        data.drop(columns=['player_id', 'team_id'], inplace=True)

    return data

def view_combineStats(drop_id=False):
    """
    Returns a dataframe containing the combine results for all years 
    in the database. The function is used to create a view of combine
    results while replacing IDs with their respective labels.

    Args:        
        drop_id (bool): A boolean to drop the ID columns.

    Returns:
        pd.DataFrame: A dataframe containing the combine results.
    """

    # Define the base query to select the draft results
    query = """
            SELECT
                cs.season_year AS year,                                
                p.player_id AS player_id,
                p.player_name AS player,
                cs.stat_nbaBrL AS nbaBrL,
                cs.stat_nbaBrR AS nbaBrR,
                cs.stat_nbaCornL AS nbaCornL,
                cs.stat_nbaCornR AS nbaCornR,
                cs.stat_nbaTop AS nbaTop,
                cs.stat_collBrL AS collBrL,
                cs.stat_collBrR AS collBrR,
                cs.stat_collCornL AS collCornL,
                cs.stat_collCornR AS collCornR,
                cs.stat_collTop AS collTop,
                cs.stat_fiftBrL AS fiftBrL,
                cs.stat_fiftBrR AS fiftBrR,
                cs.stat_fiftCornL AS fiftCornL,
                cs.stat_fiftCornR AS fiftCornR,
                cs.stat_fiftTop AS fiftTop,
                cs.stat_drblCollBrL AS drblCollBrL,
                cs.stat_drblCollBrR AS drblCollBrR,
                cs.stat_drblCollTop AS drblCollTop,
                cs.stat_drblFiftBrL AS drblFiftBrL,
                cs.stat_drblFiftBrR AS drblFiftBrR,
                cs.stat_drblFiftTop AS drblFiftTop,
                cs.stat_otmColl AS otmColl,
                cs.stat_otmFift AS otmFift,
                cs.stat_laneAgil AS laneAgil,
                cs.stat_shtlRun AS shtlRun,
                cs.stat_threeQuartSprint AS threeQuartSprint,
                cs.stat_standVerLeap AS standVerLeap,
                cs.stat_maxVerLeap AS maxVerLeap,
                cs.stat_benchPress AS benchPress,
                cs.stat_bodyFat AS bodyFat,
                cs.stat_handLen AS handLen,
                cs.stat_handWid AS handWid,
                cs.stat_ht AS ht,
                cs.stat_htShoe AS htShoe,
                cs.stat_standReach AS standReach,
                cs.stat_wt AS wt,
                cs.stat_wingspan AS wingspan
            FROM 
                combineStats cs            
            LEFT JOIN 
                players p ON cs.player_id = p.player_id            
            """
    
    # Execute the query
    data = pd.read_sql(query, engine)    

    # Order by year then by team then by player
    data.sort_values(by=['year', 'player'], inplace=True)

    # Drop unnecessary columns
    if drop_id:
        data.drop(columns=['player_id'], inplace=True)

    return data


def view_draftStats(list_id=None, drop_id=False):
    """
    Returns a dataframe containing all NCAA season statistics and
    the first NBA season statistics for all players in the draftHistory
    table. The function is used to create compile, at the moment of 
    the draft, data from the past (NCAA), preent (draft result), and
    future (NBA) of the players in a single table.

    Args:        

    Returns:
        pd.DataFrame: A dataframe containing the draft statistics.
    """

    # Create an inspector object
    inspector = sa.inspect(engine)

    # Check if draftHistory view exists
    if 'draftHistory' not in inspector.get_table_names():
        view_draftHistory()

    # Store the initial draft data
    draft_data = pd.read_sql("""SELECT * FROM draftHistory""", engine)

    # Select only the player_id present in list_id
    if list_id:
        draft_data = draft_data[draft_data['player_id'].isin(list_id)]

    # Get the unique player IDs
    player_ids = draft_data['player_id'].unique().tolist()

    # Check if playerStats view exists
    if 'playerStats' not in inspector.get_table_names():
        players_data = view_playerStats(player_ids)
    else:
        players_data = pd.read_sql('SELECT * FROM playerStats', engine)
        
        # Filter the data
        players_data = players_data[players_data['player_id'].isin(player_ids)]
    
    # Iterate over each row in draft_data
    for idx, row in draft_data.iterrows():
        # 1.1 Get the player ID
        player_id = row['player_id']
        
        # 1.2 Filter players_data for that ID
        player_seasons = players_data[players_data['player_id'] == player_id]
        
        # 1.3 Sort the results by ascending year
        player_seasons = player_seasons.sort_values(by='season_year', ascending=True)
        
        # 1.4 Count the number of years that player has
        ncaa_seasons = len(player_seasons)
        
        # 1.5 Add the number of years to that player's draft_data['ncaa_seasons']
        draft_data.at[idx, 'ncaa_seasons'] = ncaa_seasons
        
        # 1.6 Get all names from columns 3 to the end of the players_data (names list)
        names_list = player_seasons.columns[3:].tolist()
        
        # 1.7 Iterate over each year that player appears in players_data
        for i, season_row in player_seasons.iterrows():
            # 1.7.1 Select the first year row and store it in a temp df
            temp_df = season_row.to_frame().T
            
            # 1.7.2 Select only the columns present in the names list
            temp_df = temp_df[names_list]
            
            # 1.7.3 Compare the first year with the year in draft_data
            draft_year = row['year']
            season_year = season_row['season_year']
            
            # 1.7.4 Calculate the lag (draft year - season year)
            lag = draft_year - season_year
            
            # 1.7.5 Rename all columns from {old name} to {old name}_{lag}
            temp_df.columns = [f"{col}_{lag}" for col in temp_df.columns]
            
            # 1.7.6 Check if the new column names exist in draft_data, create the column if not
            for col in temp_df.columns:
                if col not in draft_data.columns:
                    draft_data[col] = None
            
            # 1.7.7 Add values from temp df to their respective columns in the draft_data row
            for col in temp_df.columns:
                draft_data.at[idx, col] = temp_df[col].values[0]


    
