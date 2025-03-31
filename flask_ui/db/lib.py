"""
DATABASE LIBRARY.

Module containing library functions to be exported when interacting with a database.
Different from 'queries' in that the library functions do not perform queries on a
database, but rather are used in operations surrounding the config, creation, deletion,
etc of a database.
"""

from flask import Flask
from flask_sqlalchemy import SQLAlchemy

from flask_ui.db.blueprint import db_app


def create_database(app: Flask, db: SQLAlchemy, db_config_uri: str):
    """
    CREATE DATABASE.

    Initializes & creates a database, integrated with a Flask app
    Input: Flask Application, SQLAlchemy Database, DB URI.
    """
    app.register_blueprint(db_app)
    app.config["SQLALCHEMY_DATABASE_URI"] = db_config_uri
    db.init_app(app)
    with app.app_context():
        # db.drop_all() # Removes all tables in the db
        db.create_all()  # Creates tables if they do not already exist
