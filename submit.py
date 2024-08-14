"""Programmatic access to the ClassifAI API."""

import argparse

import requests
from fastapi import UploadFile

# url = "http://127.0.0.1:8000/soc"
url = "https://classifai-sandbox.nw.r.appspot.com/soc"


def run(file: UploadFile):
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--file", required=True, type=str, help="Input survey data (csv)"
    )
    args = vars(parser.parse_args())

    run(file=args["file"])
