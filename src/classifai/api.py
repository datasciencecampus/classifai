"""Tools to add API functionality."""

import csv

import toml

from classifai.embedding import EmbeddingHandler


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

    def __init__(self, input_filepath: str = "data/example_survey_data.csv"):
        self.instantiate_vector_store()
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

    def instantiate_vector_store(self):
        """Connect to vector store."""
        self.embed = EmbeddingHandler(k_matches=3)

        self.embed.embed_index(
            file_name="data/soc-index/soc_title_condensed.txt"
        )

    def classify_input(self, input_data: str):
        """Classify input data in terms of survey data.

        Parameters
        ----------
        input_data : str
            Filepath to input survey data.

        Returns
        -------
        result : dict
            Dictionary of most closely related roles.
        """
        result = self.embed.search_index(
            input_data=input_data,
            id_field="id",
            embedded_fields=["job_title", "company"],
            process_output=True,
        )

        return result

    # def simplify_output(self, data: dict) -> dict:
    #     """Filter nested fields to those in config template.

    #     Parameters
    #     ----------
    #     data : dict
    #         Classified survey data as dictionary.

    #     Returns
    #     -------
    #     output : dict
    #         Filtered and nested dictionary with req'd keys only.
    #     """

    #     fields = self._config["all"]["fields"] + self._config["soc"]["fields"]
    #     output = {}
    #     for entry in data:
    #         output.update({entry["uid"]: {key: entry[key] for key in fields}})

    #     return output
