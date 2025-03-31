"""
TEST DB VIEWS.

Modulke containing tests for the views which interact with the DB
"""

import pytest

from flask_ui.app import app, db
from flask_ui.db.queries import (
    create_job_data,
    create_session,
    get_or_create_user,
)
from src.classifai.config import Config


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


def test_post_session_success(client, mock_user):
    """Test successful session creation without using mock.patch."""
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
    config = Config("UI")

    # Override `config.env_type` directly
    original_env_type = config.env_type
    config.env_type = "local"

    # Temporarily override function behavior
    original_get_user = get_or_create_user
    original_create_session = create_session
    original_create_job_data = create_job_data

    try:
        # Fake implementations to replace actual database calls
        def fake_get_local_user_credentials():
            return mock_user

        def fake_get_or_create_user(db, user_credentials):
            return user_credentials  # Simulating a User object

        def fake_create_session(db, user, session_id):
            return {
                "user": user,
                "session_id": session_id,
            }  # Fake session object

        def fake_create_job_data(db, session, job_data):
            return [
                {"job_id": job["id"], "session_id": session["session_id"]}
                for job in job_data
            ]

        # Assigning fake functions to override actual implementations
        globals()["get_or_create_user"] = fake_get_or_create_user
        globals()["create_session"] = fake_create_session
        globals()["create_job_data"] = fake_create_job_data

        response = client.post("/post_session", json=[session_id, job_data])

        assert response.status_code == 200

    finally:
        # Restore original values after the test
        config.env_type = original_env_type
        globals()["get_or_create_user"] = original_get_user
        globals()["create_session"] = original_create_session
        globals()["create_job_data"] = original_create_job_data
