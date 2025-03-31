"""
DB APP BLUEPRINT.

This module holds the views & registers a blueprint for the database application.
"""

import logging

from flask import Blueprint, Response, jsonify, request

from flask_ui.db import db
from flask_ui.db.queries import (
    create_job_data,
    create_many_results_many_jobs,
    create_session,
    get_local_user_credentials,
    get_or_create_user,
    get_recent_session_state,
    get_user,
    update_job_with_result,
    update_many_jobs_with_job_data,
)

db_app = Blueprint("db_app", __name__)


@db_app.route("/post_session", methods=["POST"])
def post_session():
    """
    POST SESSION VIEW.

    Takes a payload of 'sessionID' & 'jobsData' from the frontend.
    Creates a Session Instance & many Job instances
    """
    logging.info("POST SESSION  CALLED")
    session_id, job_data = request.json
    user_credentials = get_local_user_credentials()
    user = get_or_create_user(db, user_credentials)
    session = create_session(db, user, session_id)
    create_job_data(db, session, job_data)
    logging.info("SESSION & JOBS CREATED SUCCESFULLY")
    return Response(status=200)


@db_app.route("/post_results", methods=["POST"])
def post_results():
    """
    POST RESULTS VIEW.

    Takes a payload of 'sessionID' & 'resultsData' from the frontend
    """
    logging.info("POST RESULTS CALLED")
    session_id, results_data = request.json
    create_many_results_many_jobs(db, session_id, results_data)
    return Response(status=200)


@db_app.route("/update_job_code", methods=["PUT"])
def update_job_data():
    """
    UPDATE JOB DATA VIEW.

    Takes a payload of 'sessionID', 'jobID', and 'result' from the frontend, updating a job
    with the values from the result
    """
    logging.info("UPDATE JOB CALLED")
    session_id, job_payload = request.json
    updated_job = update_job_with_result(
        db, session_id, int(job_payload["jobId"]), job_payload["result"]
    )
    if updated_job:
        logging.info(
            f"JOB UPDATED SUCCESFULLY {updated_job, updated_job.input_id, updated_job.code}"
        )
        return Response(status=200)
    else:
        logging.info(f"JOB COULD NOT BE FOUND  {job_payload}")
        return Response(status=404)


@db_app.route("/update_many_jobs", methods=["PUT"])
def update_many_jobs():
    """
    UPDATE MANY JOBS VIEW.

    Takes a payload of 'sessionID' &  'jobsData' from the frontend, updating the mapped
    job instances with the new data.
    """
    logging.info("UPDATE MANY JOBS CALLED")
    session_id, jobs_data = request.json
    try:
        update_many_jobs_with_job_data(db, session_id, jobs_data)
        logging.info("UPDATED JOBS SUCCESFULLY")
        return Response(status=200)
    except Exception:
        logging.info("JOBS NOT UDPATED SUCCESFULLY")
        return Response(status=500)


@db_app.route("/get_previous_session", methods=["GET"])
def fetch_previous_session():
    """
    FETCH PREVIOUS SESSION VIEW.

    Takes a request from the user and returns a json payload representing the state of
    their most recent session, containing 'sessionID', 'jobsData' & 'resultsData'.
    """
    logging.info("FETCH PREVIOUS SESSION CALLED.")
    try:
        user = get_user(db, get_local_user_credentials())
        session_id, jobs_data, results_data = get_recent_session_state(
            db, user
        )
        return jsonify(session_id, jobs_data, results_data)
    except Exception:
        logging.info("UNABLE TO RETRIEVE PREVIOUS SESSION")
        return Response(status=500)
