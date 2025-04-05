# =============================================================================
# Script Name: models.py
# =============================================================================
"""
Purpose:
--------------------------------------------------------------------------------
This module defines the ORM (Object-Relational Mapping) models for the project.
Each class represents a table in the database, with relationships and data
structure defined according to the schema required for the project.

Structure:
--------------------------------------------------------------------------------
1. ORM Class Definitions - Each class corresponds to a table in the database.
    Cities:             Represents the cities where teams are based.
    Divisions:          Represents the divisions in which teams are organized.
    Conferences:        Represents the conferences to which divisions belong.
    ConferenceHistory:  Represents historical data of conferences.
    Teams:              Represents the teams participating in the league.
    TeamHistory:        Represents the historical data of teams, including 
                        changes in conference, division, and city.
    Players:            Represents the players, including their physical 
                        statistics.
    DraftHistory:       Represents historical data of player drafts.
    CombineStats:       Represents the combine statistics for players.
    PlayerHistory:      Represents the historical performance data of players.


Usage:
--------------------------------------------------------------------------------
This module is imported by database.py and any other module that requires access
to the database schema. The ORM classes are used to interact with the database,
allowing for querying and updating data using Python objects.

Dependencies:
--------------------------------------------------------------------------------
Ensure all dependencies listed in requirements.txt are installed. This module
relies on SQLAlchemy for ORM functionality.

Author:
--------------------------------------------------------------------------------
Roberto Primo Curti
[2024-07-31]
"""

#################################### Imports ####################################
import sqlalchemy as sa
from sqlalchemy.orm import relationship, declarative_base

################################## Environment ##################################
# Variables
Base = declarative_base()

################################## ORM Classes ##################################
class Cities(Base):
    __tablename__ = 'cities'
    
    geo_id = sa.Column('geo_id', sa.Integer, primary_key=True)
    city_name = sa.Column('city_name', sa.String)
    state_name = sa.Column('state_name', sa.String)
    
    # Define the relationship back to TeamHistory
    team_histories = relationship("TeamHistory", back_populates="city")

class Divisions(Base):
    __tablename__ = 'divisions'
    
    division_id = sa.Column('division_id', sa.Integer, primary_key=True)
    division_name = sa.Column('division_name', sa.String)
    
    # Define the relationship back to TeamHistory
    team_histories = relationship("TeamHistory", back_populates="division")

class Conferences(Base):
    __tablename__ = 'conferences'
    
    conference_id = sa.Column('conference_id', sa.Integer, primary_key=True)
    conference_name = sa.Column('conference_name', sa.String)
    
    # Define the relationships back to ConferenceHistory and TeamHistory
    conference_histories = relationship("ConferenceHistory", back_populates="conference")
    team_histories = relationship("TeamHistory", back_populates="conference")

class ConferenceHistory(Base):
    __tablename__ = 'conferenceHistory'

    historical_id = sa.Column('historical_id', sa.Integer, primary_key=True)
    conference_id = sa.Column('conference_id', sa.Integer, sa.ForeignKey('conferences.conference_id'))
    conference_name = sa.Column('conference_name', sa.String)
    start_year = sa.Column('start_year', sa.Integer)
    end_year = sa.Column('end_year', sa.Integer)
    
    # Define the relationship to Conferences
    conference = relationship("Conferences", back_populates="conference_histories")

class Teams(Base):
    __tablename__ = 'teams'
    
    team_id = sa.Column('team_id', sa.Integer, primary_key=True)    
    team_name = sa.Column('team_name', sa.String)
    school_name = sa.Column('school_name', sa.String)
    
    # Define the relationships back to TeamHistory, Draft, and PlayerHistory
    team_histories = relationship("TeamHistory", back_populates="team")
    draft_histories = relationship("DraftHistory", back_populates="team")
    player_histories = relationship("PlayerHistory", back_populates="team")

class TeamHistory(Base):
    __tablename__ = 'teamHistory'
    
    historical_id = sa.Column('historical_id', sa.Integer, primary_key=True)
    team_id = sa.Column('team_id', sa.Integer, sa.ForeignKey('teams.team_id'))    
    conference_id = sa.Column('conference_id', sa.Integer, sa.ForeignKey('conferences.conference_id'))
    division_id = sa.Column('division_id', sa.Integer, sa.ForeignKey('divisions.division_id'))
    city_id = sa.Column('city_id', sa.Integer, sa.ForeignKey('cities.geo_id'))
    historical_name = sa.Column('historical_name', sa.String)
    team_tag = sa.Column('team_tag', sa.String)
    start_year = sa.Column('start_year', sa.Integer)
    end_year = sa.Column('end_year', sa.Integer)
    
    # Define the relationships to Teams, Conferences, Divisions, and cities
    team = relationship("Teams", back_populates="team_histories")
    conference = relationship("Conferences", back_populates="team_histories")
    division = relationship("Divisions", back_populates="team_histories")
    city = relationship("Cities", back_populates="team_histories")

class Players(Base):
    __tablename__ = 'players'
    
    player_id = sa.Column('player_id', sa.Integer, primary_key=True)
    player_name = sa.Column('player_name', sa.String)
    stat_ht = sa.Column('stat_ht', sa.Float)
    stat_wt = sa.Column('stat_wt', sa.Float)
    
    # Define the relationships back to Draft, CombineStats, and PlayerHistory
    draft_histories = relationship("DraftHistory", back_populates="player")
    combine_stats = relationship("CombineStats", back_populates="player")
    player_histories = relationship("PlayerHistory", back_populates="player")

class DraftHistory(Base):
    __tablename__ = 'draftHistory'
    
    historical_id = sa.Column('historical_id', sa.Integer, primary_key=True)
    season_year = sa.Column('season_year', sa.Integer)
    team_id = sa.Column('team_id', sa.Integer, sa.ForeignKey('teams.team_id'))    
    player_id = sa.Column('player_id', sa.Integer, sa.ForeignKey('players.player_id'))
    draft_round = sa.Column('draft_round', sa.Integer)
    draft_pickRound = sa.Column('draft_pickRound', sa.Integer)
    draft_pickOverall = sa.Column('draft_pickOverall', sa.Integer)
    
    # Define the relationships to Teams and Players
    team = relationship("Teams", back_populates="draft_histories")
    player = relationship("Players", back_populates="draft_histories")

class CombineStats(Base):
    __tablename__ = 'combineStats'
    
    historical_id = sa.Column('historical_id', sa.Integer, primary_key=True)
    season_year = sa.Column('season_year', sa.Integer)
    player_id = sa.Column('player_id', sa.Integer, sa.ForeignKey('players.player_id'))    
    stat_nbaBrL = sa.Column('stat_nbaBrL', sa.Float)
    stat_nbaBrR = sa.Column('stat_nbaBrR', sa.Float)
    stat_nbaCornL = sa.Column('stat_nbaCornL', sa.Float)
    stat_nbaCornR = sa.Column('stat_nbaCornR', sa.Float)
    stat_nbaTop = sa.Column('stat_nbaTop', sa.Float)
    stat_collBrL = sa.Column('stat_collBrL', sa.Float)
    stat_collBrR = sa.Column('stat_collBrR', sa.Float)
    stat_collCornL = sa.Column('stat_collCornL', sa.Float)
    stat_collCornR = sa.Column('stat_collCornR', sa.Float)
    stat_collTop = sa.Column('stat_collTop', sa.Float)
    stat_fiftBrL = sa.Column('stat_fiftBrL', sa.Float)
    stat_fiftBrR = sa.Column('stat_fiftBrR', sa.Float)
    stat_fiftCornL = sa.Column('stat_fiftCornL', sa.Float)
    stat_fiftCornR = sa.Column('stat_fiftCornR', sa.Float)
    stat_fiftTop = sa.Column('stat_fiftTop', sa.Float)
    stat_drblCollBrL = sa.Column('stat_drblCollBrL', sa.Float)
    stat_drblCollBrR = sa.Column('stat_drblCollBrR', sa.Float)
    stat_drblCollTop = sa.Column('stat_drblCollTop', sa.Float)
    stat_drblFiftBrL = sa.Column('stat_drblFiftBrL', sa.Float)
    stat_drblFiftBrR = sa.Column('stat_drblFiftBrR', sa.Float)
    stat_drblFiftTop = sa.Column('stat_drblFiftTop', sa.Float)
    stat_otmColl = sa.Column('stat_otmColl', sa.Float)
    stat_otmFift = sa.Column('stat_otmFift', sa.Float)
    stat_laneAgil = sa.Column('stat_laneAgil', sa.Float)
    stat_shtlRun = sa.Column('stat_shtlRun', sa.Float)
    stat_threeQuartSprint = sa.Column('stat_threeQuartSprint', sa.Float)
    stat_standVerLeap = sa.Column('stat_standVerLeap', sa.Float)
    stat_maxVerLeap = sa.Column('stat_maxVerLeap', sa.Float)
    stat_benchPress = sa.Column('stat_benchPress', sa.Integer)
    stat_bodyFat = sa.Column('stat_bodyFat', sa.Float)
    stat_handLen = sa.Column('stat_handLen', sa.Float)
    stat_handWid = sa.Column('stat_handWid', sa.Float)
    stat_ht = sa.Column('stat_ht', sa.Float)
    stat_htShoe = sa.Column('stat_htShoe', sa.Float)
    stat_standReach = sa.Column('stat_standReach', sa.Float)
    stat_wt = sa.Column('stat_wt', sa.Float)
    stat_wingspan = sa.Column('stat_wingspan', sa.Float)
    
    # Define the relationship to Players
    player = relationship("Players", back_populates="combine_stats")

class PlayerHistory(Base):
    __tablename__ = 'playerHistory'

    historical_id = sa.Column('historical_id', sa.Integer, primary_key=True)
    season_year = sa.Column('season_year', sa.Integer)
    season_stage = sa.Column('season_stage', sa.String)
    team_id = sa.Column('team_id', sa.Integer,  sa.ForeignKey('teams.team_id'))
    player_id = sa.Column('player_id', sa.Integer,  sa.ForeignKey('players.player_id'))
    stat_games = sa.Column('stat_games', sa.Integer)
    stat_gamesStart = sa.Column('stat_gamesStart', sa.Integer)
    stat_minutes = sa.Column('stat_minutes', sa.Integer)
    stat_fg = sa.Column('stat_fg', sa.Integer)
    stat_fga = sa.Column('stat_fga', sa.Integer)
    stat_fgPct = sa.Column('stat_fgPct', sa.Float)
    stat_fg3 = sa.Column('stat_fg3', sa.Integer)
    stat_fg3a = sa.Column('stat_fg3a', sa.Integer)
    stat_fg3Pct = sa.Column('stat_fg3Pct', sa.Float)
    stat_fg2 = sa.Column('stat_fg2', sa.Integer)
    stat_fg2a = sa.Column('stat_fg2a', sa.Integer)
    stat_fg2Pct = sa.Column('stat_fg2Pct', sa.Float)
    stat_efgPct = sa.Column('stat_efgPct', sa.Float)
    stat_ft = sa.Column('stat_ft', sa.Integer)
    stat_fta = sa.Column('stat_fta', sa.Integer)
    stat_ftPct = sa.Column('stat_ftPct', sa.Float)
    stat_orb = sa.Column('stat_orb', sa.Integer)
    stat_drb = sa.Column('stat_drb', sa.Integer)
    stat_trb = sa.Column('stat_trb', sa.Integer)
    stat_ast = sa.Column('stat_ast', sa.Integer)
    stat_stl = sa.Column('stat_stl', sa.Integer)
    stat_blk = sa.Column('stat_blk', sa.Integer)
    stat_tov = sa.Column('stat_tov', sa.Integer)
    stat_pf = sa.Column('stat_pf', sa.Integer)
    stat_pts = sa.Column('stat_pts', sa.Integer)
    
    # Define the relationships to Teams and Players
    team = relationship("Teams", back_populates="player_histories")
    player = relationship("Players", back_populates="player_histories")
