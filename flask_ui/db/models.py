"""Database Models."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import DateTime, ForeignKey, String, Uuid
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    """
    Base class.

    Ensures the models can be mapped in a declarative manner:
    https://docs.sqlalchemy.org/en/20/orm/mapping_styles.html#orm-declarative-mapping
    """

    pass


"""
Initialization of the database object.

Uses flask_sqlalchemy's `SQLAlchemy` class to create a database object which can then
be initialized as config in our Flask app.
"""
db = SQLAlchemy(model_class=Base)


class User(db.Model):
    """
    User Model.

    Representation of a User of this application.
    Using an autoincremented integer id opposed to the google_id to allow for easier joining.
    """

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    google_id: Mapped[str] = mapped_column(unique=True)
    google_email: Mapped[str]

    # List of all JobsData inputs by the User. Many JobsData, 1 User
    sessions: Mapped[list["Session"]] = relationship(back_populates="user")


class Session(db.Model):
    """
    Session Model.

    Representation of a user's session, with the sessionID as the primary key.
    Acts as the parent of a set of jobs.
    Mapped directly to a User.
    """

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )  # timestamps on creation
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=func.now(),
        onupdate=func.now(),
    )  # timestamp on update
    user_id: Mapped[int] = mapped_column(ForeignKey("user.id"))

    # User who input the JobsData. 1 User, many Jobs
    user: Mapped["User"] = relationship(back_populates="sessions")

    # List of jobs contained as input within the Session. Many jobs, 1 Session
    jobs: Mapped[list["Job"]] = relationship(back_populates="session")

    # List of results tied to the jobs within the session. Many Results, 1 Session
    results: Mapped[list["Result"]] = relationship(back_populates="session")


class Job(db.Model):
    """
    Job Model.

    Representation of a job field -- a row within 'jobsData' on the frontend.
    The pk is an auto-incrementing integer instead of the input_id; this is because
    the same User may upload the same job twice: we would want these to be viewed as
    2 different jobs, as they are tied to 2 different Session inputs.

    Jobs are the children of a Session, with a foreign key relationship to a session.
    """

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    input_id: Mapped[int]
    session_id: Mapped[UUID] = mapped_column(ForeignKey("session.id"))
    description: Mapped[str] = mapped_column(String(255))
    description_orig: Mapped[str] = mapped_column(String(255))
    code: Mapped[Optional[int]]
    code_description: Mapped[Optional[str]] = mapped_column(String(255))
    code_score: Mapped[Optional[float]]
    code_rank: Mapped[Optional[int]]

    # The parent of the Job; the input the job was tied to. 1 Session, many Jobs
    session: Mapped["Session"] = relationship(back_populates="jobs")

    # List of results associated with the job, fed from the api. Many Results, 1 Job
    results: Mapped[list["Result"]] = relationship(back_populates="job")


class Result(db.Model):
    """
    Result Model.

    Representation of a specific row within a 'ResultsData' object.
    Mapped directly to the Job it was tied to through a foreign key relationship.
    """

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    input_id: Mapped[int] = mapped_column(ForeignKey("job.id"))
    session_id: Mapped[UUID] = mapped_column(ForeignKey("session.id"))
    bridge: Mapped[Optional[str]]
    description: Mapped[str] = mapped_column(String(255))
    distance: Mapped[float]
    label: Mapped[int]
    rank: Mapped[int]

    # Job the result is tied to. 1 Job, many Results
    job: Mapped["Job"] = relationship(back_populates="results")

    # Session which the results are tied too. 1 Session, many Results
    session: Mapped["Session"] = relationship(back_populates="results")
