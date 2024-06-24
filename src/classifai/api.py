"""Tools to add API functionality."""

import csv
import random  # temp for toy classifier

import toml


class API:
    """
    Class of methods controlling input of survey data.

    Parameters
    ----------
    input_filepath : str
        Relative filepath to input csv data.
        Defaults to 'data/lfs_mock.csv'

    Notes
    -----
    Currently, assumptions are made about the fields in input data and
    desirable fields in output data. The `classify_input` method is
    provided temporarily to illustrate the full end-to-end workflow.

    """

    # TODO: consider load_config class method
    _config = toml.load("config.toml")

    def __init__(self, input_filepath: str = "data/lfs_mock.csv"):
        self.input_filepath = input_filepath

    def jsonify_input(self) -> dict:
        """Convert csv at filepath to dictionary.

        Returns
        -------
        data : dict
            Dictionary representation of input csv data.
        """

        with open(self.input_filepath, "r") as file:
            reader = csv.DictReader(file)
            data = [row for row in reader]

        return data

    @staticmethod
    def classify_input(data: dict) -> dict:
        """Toy classifier pending actual classification module.

        Parameters
        ----------
        data : dict
            Dictionary of input survey data.

        Returns
        -------
        data : dict
            Input dictionary with additional keys.
        """

        for entry in data:
            entry["label"] = random.randint(1, 5)
            entry["distance"] = random.uniform(0, 1)

        return data

    def simplify_output(self, data: dict) -> dict:
        """Filter nested fields to those in config template.

        Parameters
        ----------
        data : dict
            Classified survey data as dictionary.

        Returns
        -------
        output : dict
            Filtered and nested dictionary with req'd keys only.
        """

        fields = self._config["all"]["fields"] + self._config["soc"]["fields"]
        output = {}
        for entry in data:
            output.update({entry["uid"]: {key: entry[key] for key in fields}})

        return output
