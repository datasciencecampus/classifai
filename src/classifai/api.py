"""Tools to add API functionality."""

import csv
import random  # temp for toy classifier

import toml

config = toml.load("config.toml")


class API:
    """Class of methods controlling input of survey data."""

    def __init__(self):
        pass

    @classmethod
    def jsonify_input(self, data: str) -> dict:
        """Convert csv at filepath to dictionary."""
        with open(data, "r") as file:
            reader = csv.DictReader(file)
            data = [row for row in reader]

        return data

    @classmethod
    def classify_input(self, data: dict) -> dict:
        """Toy classifier pending actual classification module."""
        for entry in data:
            entry["label"] = random.randint(1, 5)
            entry["confidence_score"] = random.uniform(0, 1)

        return data

    @classmethod
    def simplify_output(self, data: dict) -> dict:
        """Filter nested fields to those in config template."""
        fields = config["all"]["fields"] + config["soc"]["fields"]
        output = {}
        for entry in data:
            output.update({entry["uid"]: {key: entry[key] for key in fields}})

        return output
