"""
DATABASE LIBRARY.

Module containing library functions to be exported when interacting with a database.
Different from 'queries' in that the library functions do not perform queries on a
database, but rather are used in operations surrounding the config, creation, deletion,
etc of a database.
"""

from flask import Flask, Request
from flask_sqlalchemy import SQLAlchemy


def create_database(app: Flask, db: SQLAlchemy, db_config_uri: str):
    """
    CREATE DATABASE.

    Initializes & creates a database, integrated with a Flask app
    Input: Flask Application, SQLAlchemy Database, DB URI.
    """
    app.config["SQLALCHEMY_DATABASE_URI"] = db_config_uri
    db.init_app(app)
    with app.app_context():
        db.drop_all()
        db.create_all()


def get_user_credentials(request: Request):
    """
    GET USER CREDENTIALS.

    Input: HttpRequest
    Output: tuple of google id & google email credentials
    """
    return (
        request.headers.get("X-Goog-Authenticated-User-ID"),
        request.headers.get("X-Goog-Authenticated-User-Email"),
    )


def get_local_user_credentials():
    """
    GET LOCAL USER CREDENTIALS.

    Output: tuple of fake user credentials to allow for local development
    """
    return ("localID123", "localuser@mail.com")
