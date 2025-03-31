"""
Library Module.

To contain library functions which are exported and used in the app.py script.
"""

import sys

sys.path.append("src/")

from flask import Flask
from flask_sqlalchemy import SQLAlchemy

from classifai.config import Config
from flask_ui.db.lib import create_database


def remove_asterisk_labels(data):
    """Remove any ranked soc codes from the ranking.

    Parameters
    ----------
    data: dict
        the json object / python dict returned by a succeful call to FastAPI SIC or SOC endpoint
    """

    # for each rows returned ranking, iterate through the ranking and remove ranked items where the label value is *
    for entry in data["data"]:
        entry["response"] = [
            response
            for response in entry["response"]
            if response["label"] != "*"
        ]

    # do the same for the deduplicated data
    for entry in data["deduplicated_data"]:
        entry["response"] = [
            response
            for response in entry["response"]
            if response["label"] != "*"
        ]

    return data


def create_app(
    app: Flask, app_config: Config, db: SQLAlchemy, db_config_uri: str
):
    """
    CREATE APP.

    Creates a Flask application, with a conditional checking if the environment is
    local and initializing a database with the app if true.

    Input: Flask Application, App Config object, FlaskSQLAlchemy DB, DB CONFIG URI
    Output: Flask Application
    """
    if app_config.env_type == "local":
        create_database(app, db, db_config_uri)
        return app
    else:
        return app
