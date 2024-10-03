"""Programmatic access to the ClassifAI API."""

import argparse
import os

from dotenv import load_dotenv
from fastapi import UploadFile
from gcp_iap_auth.user import User, UserIAPClient

load_dotenv()

OAUTH_CLIENT_ID = os.getenv("OAUTH_CLIENT_ID")
OAUTH_CLIENT_SECRET = os.getenv("OAUTH_CLIENT_SECRET")

APP_URL = os.getenv("APP_URL")

CREDENTIAL_PATH = os.getenv("CREDENTIAL_PATH")


def run(file: UploadFile):
    """Process data on live API and write to 'outputs' folder.

    Parameters
    ----------
    file : UploadFile
        User-input csv data.
    """
    user = User(
        oauth_client_id=OAUTH_CLIENT_ID,
        oauth_client_secret=OAUTH_CLIENT_SECRET,
        credentials_path=CREDENTIAL_PATH,
    )

    iap_client = UserIAPClient(user=user)

    files = {"file": open(file, "rb")}

    response = iap_client.request(
        url=APP_URL,
        method="POST",
        files=files,
    )
    with open("outputs/example.json", "w") as f:
        f.write(response.text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--file", required=True, type=str, help="Input survey data (csv)"
    )
    args = vars(parser.parse_args())

    run(file=args["file"])
