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

    def _instantiate_vector_store(self):
        """Connect to vector store."""
        self.embed = EmbeddingHandler(k_matches=3)

        self.embed.embed_index(
            file_name="data/soc-index/soc_title_condensed.txt"
        )

    def classify_input(self, input_data: dict, embedded_fields: list):
        """Classify input data in terms of survey data.

        Parameters
        ----------
        input_data : dict
            Dictionary of input survey data.
        embedded_fields : list, optional
            The list of fields to embed and search against the database, by default None.

        Returns
        -------
        result : dict
            Dictionary of most closely related roles.
        """
        self._instantiate_vector_store()
        result = self.embed.search_index(
            input_data=input_data, embedded_fields=embedded_fields
        )

        return result

    @staticmethod
    def simplify_output(
        output_data: dict, input_data: list[dict], id_field: str
    ):
        """Process the output from the embedding search.

        Parameters
        ----------
        output_data : dict
            The output from classify input.
        input_data : list[dict]
            The input survey data read using jsonify_input.
        id_field : str
            The name of the id field.

        Returns
        -------
        output_dict: dict
            The processed result from the embedding search.
        """

        output_dict = dict()
        for label_list, description_list, distance_list, input_dict in zip(
            output_data["metadatas"],
            output_data["documents"],
            output_data["distances"],
            input_data,
        ):
            for label, description, distance in zip(
                label_list, description_list, distance_list
            ):
                label.update({"description": description})
                label.update({"distance": distance})
            output_dict[input_dict[id_field]] = label_list

        return output_dict
