"""Tools to add API functionality."""

import toml

config = toml.load("config.toml")


class Outputs:
    """Class of methods shaping output data and format."""

    def __init__(self):
        pass

    @classmethod
    def simplify_output(self, data: dict) -> dict:
        """Filter nested fields to those in config template."""
        fields = config["all"]["fields"] + config["soc"]["fields"]
        output = {}
        for uid, value in data.items():
            output.update({uid: {key: value[key] for key in fields}})

        return output
