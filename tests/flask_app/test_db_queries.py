"""
TEST DB QUERIES.

Module containing the logic for setting up the Database fixtures & then testing the db queries
from db.queries.py.
"""

from uuid import uuid4

import pytest
from flask import Flask

from flask_ui.db import db
from flask_ui.db.lib import create_database
from flask_ui.db.queries import (
    create_job,
    create_job_data,
    create_many_results_many_jobs,
    create_many_results_one_job,
    create_session,
    get_local_user_credentials,
    get_or_create_user,
    update_job_with_job_data,
    update_job_with_result,
    update_many_jobs_with_job_data,
)


@pytest.fixture(scope="module")
def test_app():
    """Creates a Flask test app and initializes the database."""
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    create_database(app, db, app.config["SQLALCHEMY_DATABASE_URI"])

    with app.app_context():
        yield app


@pytest.fixture
def test_db(test_app):
    """Provides a clean database session for each test."""
    with test_app.app_context():
        db.drop_all()
        db.create_all()
        yield db
        db.session.rollback()


@pytest.fixture
def test_user(test_db):
    """Creates a test user using local credentials."""
    user_credentials = get_local_user_credentials()
    return get_or_create_user(test_db, user_credentials)


@pytest.fixture
def test_session(test_db, test_user):
    """Creates a session linked to a test user."""
    return create_session(test_db, test_user, str(uuid4()))


def test_get_or_create_user(test_db):
    """Test user retrieval and creation."""
    user_credentials = get_local_user_credentials()

    user = get_or_create_user(test_db, user_credentials)
    assert user is not None
    assert user.google_id == user_credentials[0]

    # Ensure user is retrieved, not created again
    existing_user = get_or_create_user(test_db, user_credentials)
    assert existing_user.id == user.id


def test_create_session(test_db, test_user):
    """Test session creation linked to a user."""
    session_id = uuid4()
    session = create_session(test_db, test_user, session_id)

    assert session is not None
    assert session.user_id == test_user.id


def test_create_job(test_db, test_session):
    """Test job creation linked to a session."""
    job_data = {
        "id": "1",
        "description": "Test Job",
        "description_orig": "Original Desc",
        "code": "3957",
        "code_description": "Test Code Desc",
        "code_score": "0.015",
        "code_rank": "5",
    }

    job = create_job(test_db, test_session, job_data)
    test_db.session.commit()

    assert job is not None
    assert job.session_id == test_session.id
    assert job.description == "Test Job"


def test_create_job_data(test_db, test_session):
    """Test bulk job creation."""
    jobs_data = [
        {
            "id": "1",
            "description": "Job 1",
            "description_orig": "Desc 1",
            "code": "",
            "code_description": "",
            "code_score": "",
            "code_rank": "",
        },
        {
            "id": "3",
            "description": "Job 2",
            "description_orig": "Desc 2",
            "code": "3648",
            "code_description": "Code Desc 2",
            "code_score": "0.1",
            "code_rank": "12",
        },
    ]

    jobs = create_job_data(test_db, test_session, jobs_data)
    assert len(jobs) == 2
    assert jobs[0].description == "Job 1"
    assert jobs[1].description == "Job 2"


def test_create_many_results_one_job(test_db, test_session):
    """Test creating multiple results for one job."""
    job_data = {
        "id": "6",
        "description": "Test Job Results",
        "description_orig": "Original Desc",
        "code": "4839",
        "code_description": "Test Code Desc",
        "code_score": "0.5",
        "code_rank": "10",
    }

    job = create_job(test_db, test_session, job_data)
    test_db.session.commit()

    result_data_object = {
        "input_id": job.input_id,
        "response": [
            {
                "bridge": "",
                "description": "Res 1",
                "distance": "5.5",
                "label": "1",
                "rank": "2",
            },
            {
                "bridge": "",
                "description": "Res 2",
                "distance": "3.2",
                "label": "0",
                "rank": "5",
            },
        ],
    }

    results = create_many_results_one_job(
        test_db, test_session.id, result_data_object
    )
    assert len(results) == 2
    assert results[0].label == 1
    assert results[1].bridge is None


def test_create_many_results_many_jobs(test_db, test_session):
    """Test bulk creation of results for multiple jobs."""
    jobs_data = [
        {
            "id": "8",
            "description": "Job A",
            "description_orig": "Orig A",
            "code": "2849",
            "code_description": "Code A",
            "code_score": "0.00",
            "code_rank": "6",
        },
        {
            "id": "1",
            "description": "Job B",
            "description_orig": "Orig B",
            "code": "9362",
            "code_description": "Code B",
            "code_score": "0.2",
            "code_rank": "7",
        },
    ]

    create_job_data(test_db, test_session, jobs_data)

    results_array = [
        {
            "input_id": "8",
            "response": [
                {
                    "bridge": "",
                    "description": "Job A1",
                    "distance": "0.1",
                    "label": "1",
                    "rank": "1",
                },
                {
                    "bridge": "",
                    "description": "Job A2",
                    "distance": "0.00",
                    "label": "0",
                    "rank": "3",
                },
            ],
        },
        {
            "input_id": "1",
            "response": [
                {
                    "bridge": "",
                    "description": "Job B1",
                    "distance": "0.7",
                    "label": "6",
                    "rank": "2",
                },
            ],
        },
    ]

    all_results = create_many_results_many_jobs(
        test_db, test_session.id, results_array
    )
    assert len(all_results) == 2  # Two jobs
    assert len(all_results[0]) == 2  # First job has 2 results
    assert len(all_results[1]) == 1  # Second job has 1 result


def test_update_job_with_result(test_db, test_session):
    """Test updating a job with a result."""
    job_data = {
        "id": "10",
        "description": "Test Job",
        "description_orig": "Original Desc",
        "code": "1234",
        "code_description": "Initial Desc",
        "code_score": "0.1",
        "code_rank": "1",
    }
    job = create_job(test_db, test_session, job_data)
    test_db.session.commit()

    result_data = {
        "label": "5678",
        "description": "Updated Desc",
        "distance": "0.9",
        "rank": "2",
    }
    updated_job = update_job_with_result(
        test_db, test_session.id, job.input_id, result_data
    )
    test_db.session.commit()

    assert updated_job is not None
    assert updated_job.code == 5678
    assert updated_job.code_description == "Updated Desc"
    assert updated_job.code_score == 0.9
    assert updated_job.code_rank == 2


def test_update_job_with_job_data(test_db, test_session):
    """Test updating a job with new job data."""
    job_data = {
        "id": "15",
        "description": "Old Job",
        "description_orig": "Old Desc",
        "code": "4321",
        "code_description": "Old Code Desc",
        "code_score": "0.05",
        "code_rank": "3",
    }
    create_job(test_db, test_session, job_data)
    test_db.session.commit()

    new_job_data = {
        "id": "15",
        "description": "New Job Description",
        "description_orig": "New Desc",
        "code": "9999",
        "code_description": "New Code Desc",
        "code_score": "0.8",
        "code_rank": "9",
    }
    updated_job = update_job_with_job_data(
        test_db, test_session.id, new_job_data
    )
    test_db.session.commit()

    assert updated_job is not None
    assert updated_job.description == "New Job Description"
    assert updated_job.description_orig == "New Desc"
    assert updated_job.code == 9999
    assert updated_job.code_description == "New Code Desc"
    assert updated_job.code_score == 0.8
    assert updated_job.code_rank == 9


def test_update_many_jobs_with_job_data(test_db, test_session):
    """Test bulk updating multiple jobs."""
    jobs_data = [
        {
            "id": "21",
            "description": "Job 1",
            "description_orig": "Desc 1",
            "code": "1111",
            "code_description": "Code Desc 1",
            "code_score": "0.3",
            "code_rank": "4",
        },
        {
            "id": "22",
            "description": "Job 2",
            "description_orig": "Desc 2",
            "code": "2222",
            "code_description": "Code Desc 2",
            "code_score": "0.4",
            "code_rank": "5",
        },
    ]
    create_job_data(test_db, test_session, jobs_data)
    test_db.session.commit()

    updated_jobs_data = [
        {
            "id": "21",
            "description": "Updated Job 1",
            "description_orig": "Updated Desc 1",
            "code": "5555",
            "code_description": "Updated Code Desc 1",
            "code_score": "0.7",
            "code_rank": "7",
        },
        {
            "id": "22",
            "description": "Updated Job 2",
            "description_orig": "Updated Desc 2",
            "code": "6666",
            "code_description": "Updated Code Desc 2",
            "code_score": "0.9",
            "code_rank": "8",
        },
    ]
    updated_jobs = update_many_jobs_with_job_data(
        test_db, test_session.id, updated_jobs_data
    )
    test_db.session.commit()

    assert len(updated_jobs) == 2
    assert updated_jobs[0].description == "Updated Job 1"
    assert updated_jobs[0].code == 5555
    assert updated_jobs[1].description == "Updated Job 2"
    assert updated_jobs[1].code == 6666
