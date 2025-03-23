"""Tests for API class."""

import pytest

from classifai.search_result_builders import (
    condense_rankings,
    create_deduplicated_response,
    naive_scorer,
)


@pytest.mark.parametrize(
    "input_data_condense_rankings, expected_output_condense_rankings",
    [
        # Test Case 1: Normal case
        (
            [
                {
                    "input_id": "1",
                    "response": [
                        {
                            "label": "1",
                            "description": "A butcher",
                            "distance": 0.11,
                            "rank": 1,
                        },
                        {
                            "label": "2",
                            "description": "A baker",
                            "distance": 0.23,
                            "rank": 2,
                        },
                        {
                            "label": "1",
                            "description": "Pork creator",
                            "distance": 0.27,
                            "rank": 3,
                        },
                        {
                            "label": "2",
                            "description": "Loaf creator",
                            "distance": 0.52,
                            "rank": 4,
                        },
                    ],
                }
            ],
            [
                {
                    "input_id": "1",
                    "response": [
                        {
                            "label": "1",
                            "descriptions": ["A butcher", "Pork creator"],
                            "distances": [0.11, 0.27],
                            "rankings": [1, 3],
                        },
                        {
                            "label": "2",
                            "descriptions": ["A baker", "Loaf creator"],
                            "distances": [0.23, 0.52],
                            "rankings": [2, 4],
                        },
                    ],
                }
            ],
        ),
        # Test Case 2: Out of order ranking
        (
            [
                {
                    "input_id": 1,
                    "response": [
                        {
                            "label": "2",
                            "description": "Loaf creator",
                            "distance": 0.52,
                            "rank": 4,
                        },
                        {
                            "label": "2",
                            "description": "A baker",
                            "distance": 0.23,
                            "rank": 2,
                        },
                        {
                            "label": "1",
                            "description": "A butcher",
                            "distance": 0.11,
                            "rank": 1,
                        },
                        {
                            "label": "1",
                            "description": "Pork creator",
                            "distance": 0.15,
                            "rank": 3,
                        },
                    ],
                }
            ],
            [
                {
                    "input_id": "1",
                    "response": [
                        {
                            "label": "2",
                            "descriptions": ["Loaf creator", "A baker"],
                            "distances": [0.52, 0.23],
                            "rankings": [4, 2],
                        },
                        {
                            "label": "1",
                            "descriptions": ["A butcher", "Pork creator"],
                            "distances": [0.11, 0.15],
                            "rankings": [1, 3],
                        },
                    ],
                }
            ],
        ),
        # Test Case 3: Empty ranking
        (
            [
                {
                    "input_id": "1",
                    "response": [],
                }
            ],
            [
                {
                    "input_id": "1",
                    "response": [],
                }
            ],
        ),
    ],
)
def test_condense_rankings(
    input_data_condense_rankings, expected_output_condense_rankings
):
    """Main test function for condensed ranking."""
    result = condense_rankings(input_data_condense_rankings)
    assert result == expected_output_condense_rankings


@pytest.mark.parametrize(
    "input_data_create_deduplicated_response, expected_output_create_deduplicated_response",
    [
        # Test Case 1: Normal case
        (
            [
                {
                    "input_id": "1",
                    "response": [
                        {
                            "label": "1",
                            "description": "A butcher",
                            "distance": 0.11,
                            "rank": 1,
                        },
                        {
                            "label": "2",
                            "description": "A baker",
                            "distance": 0.23,
                            "rank": 2,
                        },
                        {
                            "label": "1",
                            "description": "Pork creator",
                            "distance": 0.27,
                            "rank": 3,
                        },
                        {
                            "label": "2",
                            "description": "Loaf creator",
                            "distance": 0.52,
                            "rank": 4,
                        },
                    ],
                }
            ],
            [
                {
                    "input_id": "1",
                    "response": [
                        {
                            "label": "1",
                            "description": "A butcher",
                            "distance": 0.11,
                            "rank": 1,
                            "score": "0.53",
                        },
                        {
                            "label": "2",
                            "description": "A baker",
                            "distance": 0.23,
                            "rank": 2,
                            "score": "0.47",
                        },
                    ],
                }
            ],
        ),
        # Test Case 2: Out of order ranking
        (
            [
                {
                    "input_id": 1,
                    "response": [
                        {
                            "label": "2",
                            "description": "Loaf creator",
                            "distance": 0.52,
                            "rank": 4,
                        },
                        {
                            "label": "2",
                            "description": "A baker",
                            "distance": 0.23,
                            "rank": 2,
                        },
                        {
                            "label": "1",
                            "description": "A butcher",
                            "distance": 0.11,
                            "rank": 1,
                        },
                        {
                            "label": "1",
                            "description": "Pork creator",
                            "distance": 0.15,
                            "rank": 3,
                        },
                    ],
                }
            ],
            [
                {
                    "input_id": "1",
                    "response": [
                        {
                            "label": "1",
                            "description": "A butcher",
                            "distance": 0.11,
                            "rank": 1,
                            "score": "0.53",
                        },
                        {
                            "label": "2",
                            "description": "A baker",
                            "distance": 0.23,
                            "rank": 2,
                            "score": "0.47",
                        },
                    ],
                }
            ],
        ),
        # Test Case 3: Empty ranking
        (
            [
                {
                    "input_id": "1",
                    "response": [],
                }
            ],
            [
                {
                    "input_id": "1",
                    "response": [],
                }
            ],
        ),
    ],
)
def test_create_deduplicated_response(
    input_data_create_deduplicated_response,
    expected_output_create_deduplicated_response,
):
    """Main test function for create deduplicates_response function."""
    deduplicated_response = create_deduplicated_response(
        input_data_create_deduplicated_response, naive_scorer
    )
    assert (
        deduplicated_response == expected_output_create_deduplicated_response
    )


# def test_create_deduplicated_response(input_data_cdr, naive_scorer):
#    assert 1 == 1


# Run the test with pytest
if __name__ == "__main__":
    pytest.main()
