"""
Database Queries.

Module for queries which physically interact with an existing databse.
"""

from uuid import UUID

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
    session = Session(user_id=user.id, id=UUID(session_id))
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
    input_id = int(result_data_object["input_id"])
    return db.session.execute(
        db.select(Job.id)
        .where(Job.input_id == input_id)
        .where(Job.session_id == UUID(session_id))
    ).scalar_one()


def _create_result(
    db: SQLAlchemy, session_id: UUID | str, job_id: int, result: dict
):
    new_result = Result(
        input_id=job_id,
        session_id=UUID(session_id),
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
