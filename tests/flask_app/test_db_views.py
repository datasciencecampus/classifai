"""
TEST DB VIEWS.

Module containing tests for the views which interact with the DB
"""
# ruff: noqa

import sys
from unittest import mock

sys.path.append(".")
sys.path.append("src/")

import pytest

pytestmark = pytest.mark.skip(
    reason="Importing the app fails because of authentication issues on GH actions."
)

# from flask_ui.app import app, db  # noqa


@pytest.fixture
def client():
    """Mock Flask Client."""
    app.config["TESTING"] = True
    with app.app_context():
        db.drop_all()
        db.create_all()
    with app.test_client() as client:
        yield client


@pytest.fixture
def mock_user():
    """Mock user credentials."""
    return ("localID123", "localuser@mail.com")


@mock.patch("flask_ui.db.queries.create_job_data")
@mock.patch("flask_ui.db.queries.create_session")
@mock.patch("flask_ui.db.queries.get_or_create_user")
@mock.patch("classifai.config.Config")
def test_post_session_success(
    mock_config,
    mock_get_or_create_user,
    mock_create_session,
    mock_create_job_data,
    client,
    mock_user,
):
    """Test successful session creation using mock.patch."""
    session_id = "550e8400-e29b-41d4-a716-446655440000"
    job_data = [
        {
            "id": "4",
            "description": "Test Job",
            "description_orig": "Test Job Orig",
            "code": "",
            "code_description": "",
            "code_score": "",
            "code_rank": "",
        }
    ]

    # Configure mocks
    mock_config_instance = mock.MagicMock()
    mock_config_instance.env_type = "local"
    mock_config.return_value = mock_config_instance

    # Set up mock return values
    mock_get_or_create_user.return_value = mock_user
    mock_create_session.return_value = {
        "user": mock_user,
        "session_id": session_id,
    }
    mock_create_job_data.return_value = [
        {"job_id": job["id"], "session_id": session_id} for job in job_data
    ]

    # Make the request
    response = client.post("/post_session", json=[session_id, job_data])

    # Assertions
    assert response.status_code == 200

    # Verify the mocks were called appropriately
    assert mock_get_or_create_user.call_count <= 1
    assert mock_create_session.call_count <= 1
    assert mock_create_job_data.call_count <= 1
