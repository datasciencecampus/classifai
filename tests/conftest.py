"""Fixtures for the embedding module."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_query_input():
    """Return sample input query data."""

    sample_input_data = [
        {"id": 1, "job_title": "statistician", "employer": "ONS"},
        {"id": 2, "job_title": "economist", "employer": "HMT"},
        {"id": 3, "job_title": "social researcher", "description": "Analyst."},
        {},
    ]

    return sample_input_data


@pytest.fixture
def sample_query_result():
    """Return sample query output."""

    sample_query_result = {
        "ids": [["a", "b"], ["c"]],
        "distances": [[1.245, 3.456], [1.145, 4.722]],
        "metadatas": [
            [{"label": 212}, {"label": 12}],
            [{"label": 34}, {"label": 25}],
        ],
        "embeddings": None,
        "documents": [
            [
                "Environment professionals",
                "Environmental health professionals",
            ],
            ["Hospital porters", "Generalist medical practitioners"],
        ],
    }

    return sample_query_result


@pytest.fixture
def sample_query_processed():
    """Return sample query processed output."""

    sample_query_processed = {
        1: [
            {
                "label": 212,
                "description": "Environment professionals",
                "distance": 1.245,
            },
            {
                "label": 12,
                "description": "Environmental health professionals",
                "distance": 3.456,
            },
        ],
        2: [
            {
                "label": 34,
                "description": "Hospital porters",
                "distance": 1.145,
            },
            {
                "label": 25,
                "description": "Generalist medical practitioners",
                "distance": 4.722,
            },
        ],
    }

    return sample_query_processed


@pytest.fixture(scope="session")
def survey_csv(tmp_path_factory):
    """Return filepath to temp survey data."""

    path = tmp_path_factory.mktemp("data") / "lfs_test.csv"
    data = [
        {
            "id": "0017",
            "job_title": "Musician",
            "company": "Office for National Statistics",
            "miscellaneous": "I like dogs",
        }
    ]
    df = pd.DataFrame.from_dict(data)
    df.to_csv(path)

    return path


@pytest.fixture()
def soc_df_input():
    """Return sample unprocessed SOC dataframe."""

    input_soc_df = pd.DataFrame(
        dict(
            major_group=[5, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
            sub_major_group=[np.NaN, 51, np.NaN, np.NaN, np.NaN, np.NaN],
            minor_group=[np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
            unit_group=[np.NaN, 22, 2223, 2223, 1525, 1237],
            group_title=[
                "Skilled Trades",
                "Agriculture",
                "Agriculture & Related",
                "Farmers",
                np.NaN,
                np.NaN,
            ],
            sub_unit_group=[
                6615,
                "1525/23",
                "626322",
                np.NaN,
                "TYPICAL ENTRY ROUTES AND ASSOCIATED QUALIFICATIONS",
                "Other value",
            ],
            descriptions=[
                "Skiled Trades work...",
                "Agriculture is...",
                "Agriculture & Related is...",
                "Farmers...",
                "GCSEs",
                "Work is usually between 9 and 5...",
            ],
        ),
        dtype=object,
    )

    return input_soc_df


@pytest.fixture()
def soc_df_education():
    """Return clean SOC education dataframe."""

    soc_df_education = pd.DataFrame(
        dict(join_column=[1525], descriptions=["GCSEs"])
    )

    return soc_df_education


@pytest.fixture()
def soc_df_clean():
    """Return clean SOC dataframe."""

    soc_df_clean = pd.DataFrame(
        dict(
            major_group=["Skilled Trades", "Skilled Trades", "Skilled Trades"],
            sub_major_group=[np.NaN, "Agriculture", "Agriculture"],
            minor_group=[np.NaN, np.NaN, np.NaN],
            unit_group=[np.NaN, "Agriculture", "Agriculture & Related"],
            group_title=[
                "Skilled Trades",
                "Agriculture",
                "Agriculture & Related",
            ],
            sub_unit_group=[6615, "1525/23", "626322"],
            descriptions=[
                "Skiled Trades work...",
                "Agriculture is...",
                "Agriculture & Related is...",
            ],
            soc_code=[6615, 152523, 626322],
        )
    )

    return soc_df_clean
