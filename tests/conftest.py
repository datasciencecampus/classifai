"""Fixtures for the embedding module."""

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
