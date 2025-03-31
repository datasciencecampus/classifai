"""
Database Queries.

Module for queries which physically interact with an existing databse.
"""

from uuid import UUID

from flask import Request
from flask_sqlalchemy import SQLAlchemy

from flask_ui.db.models import Job, Result, Session, User


def _int_or_null(potential_int: str):
    try:
        return int(potential_int)
    except Exception:
        return None


def _float_or_null(potential_float: str):
    try:
        return float(potential_float)
    except Exception:
        return None


def _str_or_null(potential_empty_str):
    if potential_empty_str == "":
        return None
    else:
        try:
            return str(potential_empty_str)
        except Exception:
            return None


def _stringify(value):
    """
    STRINGIFY.

    Takes an int, None, float, or str.
    Checks if the input is None, returning an
    empty string, else returning a stringed value.
    """
    if value is None:
        return ""
    else:
        return str(value)


def get_user_credentials(request: Request):
    """
    GET USER CREDENTIALS.

    Input: HttpRequest
    Output: tuple of google id & google email credentials
    """
    return (
        request.headers.get("X-Goog-Authenticated-User-ID"),
        request.headers.get("X-Goog-Authenticated-User-Email"),
    )


def get_local_user_credentials():
    """
    GET LOCAL USER CREDENTIALS.

    Output: tuple of fake user credentials to allow for local development
    """
    return ("localID123", "localuser@mail.com")


def get_user(db: SQLAlchemy, user_credentials: tuple):
    """
    GET USER.

    Input: FlaskSQLAlchemy Database, Tuple of user credentials
    Output: User, or none
    """
    return db.session.execute(
        db.select(User)
        .where(User.google_id == user_credentials[0])
        .where(User.google_email == user_credentials[1])
    ).scalar_one_or_none()


def get_or_create_user(db: SQLAlchemy, user_credentials: tuple):
    """
    GET OR CREATE USER.

    Gets a User model based on user credentials, creating a user if
    none found

    Input: FlaskSQLAlchemy Database, Tuple of user credentials
    Output: User
    """
    user = get_user(db, user_credentials)
    if user:
        return user
    else:
        user = User(
            google_id=user_credentials[0], google_email=user_credentials[1]
        )
        db.session.add(user)
        db.session.commit()
        return user


def create_session(db: SQLAlchemy, user: User, session_id: UUID | str):
    """
    CREATE SESSION.

    Creates a Session Instance, linking to a User

    Input: FlaskSQLAlchemy Database, User, session_id
    Output: Session
    """
    if isinstance(session_id, str):
        session_id = UUID(session_id)  # Convert string to UUID

    session = Session(user_id=user.id, id=session_id)
    db.session.add(session)
    db.session.commit()
    return session


def create_job(db: SQLAlchemy, session: Session, job: dict):
    """
    CREATE JOB.

    Creates a Job instance, linking them to a Session instance
    NOTE: No db.session.commit() used. Data would need to be committed outside this function

    Input: FlaskSQLAlchemy Database, Session, potential job
    Output: Job
    """
    job = Job(
        input_id=int(job["id"]),
        session_id=session.id,
        description=str(job["description"]),
        description_orig=str(job["description_orig"]),
        code=_int_or_null(job["code"]),
        code_description=job["code_description"],
        code_score=_float_or_null(job["code_score"]),
        code_rank=_int_or_null(job["code_rank"]),
    )
    db.session.add(job)
    return job


def create_job_data(db: SQLAlchemy, session: Session, job_data: list[dict]):
    """
    CREATE JOB DATA.

    Creates many instances of 'Jobs'.
    This function will be used mainly when receiving a 'jobsData' payload from the frontend.

    Input: FlaskSQLAlchemy Database, Session, list of potential jobs
    Output: List of Jobs
    """
    job_result = [create_job(db, session, job) for job in job_data]
    db.session.commit()
    return job_result


def _get_job_id_from_result_data_object(
    db: SQLAlchemy, session_id: UUID | str, result_data_object: dict
):
    """
    GET JOB ID FROM RESULT.

    Gets a Job's PK from a result_data_object, posted from the frontend.
    """
    if isinstance(session_id, str):
        session_id = UUID(session_id)  # Convert string to UUID

    input_id = int(result_data_object["input_id"])
    return db.session.execute(
        db.select(Job.id)
        .where(Job.input_id == input_id)
        .where(Job.session_id == session_id)
    ).scalar_one()


def _create_result(
    db: SQLAlchemy, session_id: UUID | str, job_id: int, result: dict
):
    if isinstance(session_id, str):
        session_id = UUID(session_id)  # Convert string to UUID

    new_result = Result(
        input_id=job_id,
        session_id=session_id,
        bridge=_str_or_null(result["bridge"]),
        description=result["description"],
        distance=float(result["distance"]),
        label=int(result["label"]),
        rank=int(result["rank"]),
    )
    db.session.add(new_result)
    return new_result


def create_many_results_one_job(
    db: SQLAlchemy, session_id: UUID | str, result_data_object: dict
):
    """
    CREATE MANY RESULTS ONE JOB.

    Takes an input of a single result data object, not the full session payload, and creates many
    results from this object, tied to one job.
    """
    job_id = _get_job_id_from_result_data_object(
        db, session_id, result_data_object
    )
    results_list = []
    for result in result_data_object["response"]:
        res = _create_result(db, session_id, job_id, result)
        results_list.append(res)
    db.session.commit()
    return results_list


def create_many_results_many_jobs(
    db: SQLAlchemy, session_id: UUID | str, results_array: list[dict]
):
    """
    CREATE MANY RESULTS MANY JOBS.

    This takes a full payload from the frontend containing the full 'resultsData' array, which contains
    many 'resultsData' objects. Iterating through the array and creating results which are tied to the
    session & jobs.
    """
    return [
        create_many_results_one_job(db, session_id, result_data_object)
        for result_data_object in results_array
    ]


def _get_job_from_id(db: SQLAlchemy, session_id: str | UUID, job_id: int):
    if isinstance(session_id, str):
        session_id = UUID(session_id)

    return db.session.execute(
        db.select(Job)
        .where(Job.session_id == session_id)
        .where(Job.input_id == job_id)
    ).scalar_one_or_none()


def update_job_with_result(
    db: SQLAlchemy, session_id: str | UUID, job_id: int, result: dict
):
    """
    UPDATE JOB WITH RESULT.

    Takes a sessionID, jobID & result object from the frontend, selects an existing job form the database
    and updates it with the result object's code values.
    """
    job = _get_job_from_id(db, session_id, job_id)

    if job:
        job.code = int(result["label"])
        job.code_description = result["description"]
        job.code_score = float(result["distance"])
        job.code_rank = int(result["rank"])

        db.session.commit()
        return job

    else:
        return None


def update_job_with_job_data(
    db: SQLAlchemy, session_id: str | UUID, job_data: dict
):
    """
    UPDATE JOB WITH JOB DATA.

    Updates an existing job instance with new data
    """
    job = _get_job_from_id(db, session_id, int(job_data["id"]))

    if job:
        job.description = str(job_data["description"])
        job.description_orig = str(job_data["description_orig"])
        job.code = _int_or_null(job_data["code"])
        job.code_description = job_data["code_description"]
        job.code_score = _float_or_null(job_data["code_score"])
        job.code_rank = _int_or_null(job_data["code_rank"])

        return job
    else:
        return None


def update_many_jobs_with_job_data(
    db: SQLAlchemy, session_id: str | UUID, job_data_array: list[dict]
):
    """
    UPDATE MANY JOBS.

    Updates a list|array of existing jobs with new job data
    """
    return [
        update_job_with_job_data(db, session_id, job_data)
        for job_data in job_data_array
    ]


def _get_results_data_from_job(job: Job):
    results_data = []
    for result in job.results:
        parsed_result = {
            "bridge": _stringify(result.bridge),
            "description": result.description.strip(),
            "distance": _stringify(result.distance),
            "label": _stringify(result.label),
            "rank": _stringify(result.rank),
        }
        results_data.append(parsed_result)
    return {"input_id": _stringify(job.input_id), "response": results_data}


def _parse_jobs_to_jobs_and_results_data(jobs: list[Job]):
    jobs_data, results_data = [], []
    for job in jobs:
        parsed_job = {
            "id": _stringify(job.input_id),
            "description": _stringify(job.description),
            "description_orig": _stringify(job.description_orig),
            "code": _stringify(job.code),
            "code_description": _stringify(job.code_description),
            "code_score": _stringify(job.code_score),
            "code_rank": _stringify(job.code_rank),
        }
        jobs_data.append(parsed_job)
        results_data.append(_get_results_data_from_job(job))
    return (jobs_data, results_data)


def get_recent_session_state(db: SQLAlchemy, user: User):
    """
    GET RECENT SESSION STATE.

    Takes the argument of a db & user, returns the most recent sessionID, with
    the jobsData & resultsData associated with it. The data parsed into a format which
    can be easily posted to the frontend in json.
    """
    session = user.sessions[-1]
    if session:
        jobs_data, results_data = _parse_jobs_to_jobs_and_results_data(
            session.jobs
        )
        return (session.id, jobs_data, results_data)
    else:
        return None
