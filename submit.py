"""Programmatic access to the ClassifAI API."""

import requests
from fastapi import UploadFile

url = "http://127.0.0.1:8000/soc"


def run(file: UploadFile = "data/example_survey_data.csv"):
    """Process data on live API and write to 'outputs' folder.

    Parameters
    ----------
    file : UploadFile
        User-input csv data.
    """

    files = {"file": open(file, "rb")}
    response = requests.post(url, files=files)
    with open("outputs/example.json", "w") as f:
        f.write(response.text)


if __name__ == "__main__":
    run()
