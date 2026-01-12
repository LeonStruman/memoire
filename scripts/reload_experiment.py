import csv
import pandas as pd
from app import create_app
from flask import current_app
from app.models import User, UserPII, Log, Question, Answer
from app import db
from config import Config as Cf
from sqlalchemy import inspect
from sqlalchemy.exc import NoSuchTableError
from dotenv import load_dotenv
import os
from pathlib import Path

# Load the .env file
load_dotenv()

app = create_app()

USER_FILENAME = 'user.csv'
USERPII_FILENAME = 'userpii.csv'
QUESTION_FILENAME = 'question.csv'
ANSWER_FILENAME = 'answer.csv'


def populate_db(db_name, csv_file):
    df = pd.read_csv(csv_file, delimiter=";")
    # convert missing values in None
    df = df.astype(object)
    df = df.where(pd.notnull(df), None)
    for record in df.to_dict("records"):
        new_entry = db_name(**record)
        db.session.add(new_entry)
        db.session.commit()


def drop_tables_in_order(table_names, foreign_keys, engine):
    # Function to perform topological sort
    def topological_sort(graph):
        visited = set()
        stack = []

        def dfs(node):
            visited.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    dfs(neighbor)
            stack.append(node)

        for node in graph.keys():
            if node not in visited:
                dfs(node)
        return stack

    # Create a graph of foreign key dependencies
    dependency_graph = {table: [] for table in table_names}
    for table, fks in foreign_keys.items():
        for fk in fks:
            dependency_graph[fk['referred_table']].append(table)

    # Perform topological sort to get the correct order of dropping tables
    sorted_tables = topological_sort(dependency_graph)
    print(sorted_tables)

    # Drop tables in the sorted order
    for table_name in sorted_tables:
        if inspect(db.engine).has_table(table_name):
            db.metadata.tables[table_name].drop(db.engine)
        #engine.execute(f"DROP TABLE IF EXISTS {table_name};")


def drop_all_tables():
    """Drop tables in correct order"""
    # Get metadata
    metadata = db.metadata

    # Get all table names
    table_names = metadata.tables.keys()
    print(table_names)

    # Create a dictionary to store foreign key relationships
    foreign_keys = {}
    inspector = inspect(db.engine)
    for table_name in table_names:
        try:
            foreign_keys[table_name] = inspector.get_foreign_keys(table_name)
        except NoSuchTableError:
            foreign_keys[table_name] = []

    drop_tables_in_order(table_names, foreign_keys, db.engine)

def populate_all_db():
    data_path = Path(os.getenv("DATA_WEBAPP_PATH"))
    populate_db(User, data_path / USER_FILENAME)
    populate_db(UserPII, data_path / USERPII_FILENAME)
    populate_db(Question, data_path / QUESTION_FILENAME)
    populate_db(Answer, data_path / ANSWER_FILENAME)


def reload_databases():
    with app.app_context():
        drop_all_tables()
        db.create_all()
        populate_all_db()
        print('reloaded databases with success !')
