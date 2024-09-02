"""Tests for the doc_utils module."""

from classifai.doc_utils import (
    clean_job_description,
    clean_job_title,
    clean_text,
)

"""Tests for clean_text."""


def test_clean_text_basic():
    """Test clean_text function with basic input."""
    assert (
        clean_text("Teacher, Statistics (secondary school)")
        == "secondary school statistics teacher"
    )


def test_clean_text_no_brackets():
    """Test clean_text function with input that does not contain brackets."""
    assert clean_text("Teacher, Statistics") == "statistics teacher"


def test_clean_text_empty_string():
    """Test clean_text function with an empty string as input."""
    assert clean_text("") == ""


def test_clean_text_only_brackets():
    """Test clean_text function with input that only contains brackets."""
    assert clean_text("(secondary school)") == "secondary school"


def test_clean_text_special_characters():
    """Test clean_text function with input that contains special characters."""
    assert (
        clean_text("Teacher, Statistics (secondary: school)")
        == "secondary school statistics teacher"
    )


def test_clean_text_multiple_commas():
    """Test clean_text function with input that contains multiple commas."""
    assert (
        clean_text("Teacher, Statistics, Maths (secondary school)")
        == "secondary school maths statistics teacher"
    )


"""Tests for clean_job_tile."""


def test_clean_job_title_basic():
    """Test that 'n.e.c.' is removed from the job title."""
    assert clean_job_title("Engineer n.e.c.") == "Engineer"


def test_clean_job_title_no_nec():
    """Test that job titles without 'n.e.c.' remain unchanged."""
    assert clean_job_title("Software Developer") == "Software Developer"


def test_clean_job_title_multiple_nec():
    """Test that multiple occurrences of 'n.e.c.' are removed."""
    assert clean_job_title("Engineer n.e.c. n.e.c.") == "Engineer"


def test_clean_job_title_with_spaces():
    """Test that leading and trailing spaces are removed after 'n.e.c.' is removed."""
    assert clean_job_title("  Engineer n.e.c.  ") == "Engineer"


def test_clean_job_title_empty_string():
    """Test that an empty string remains unchanged."""
    assert clean_job_title("") == ""


def test_clean_job_title_only_nec():
    """Test that a string containing only 'n.e.c.' is cleaned to an empty string."""
    assert clean_job_title("n.e.c.") == ""


"""Tests for clean_job_description."""


def test_clean_job_description_job_holders():
    """Test clean_job_description function when job holders are mentioned in the description."""
    job_title = "Software Developer"
    job_description = (
        "Job holders in this group perform software development tasks."
    )
    expected_result = "Software Developer"
    assert clean_job_description(job_title, job_description) == expected_result


def test_clean_job_description_no_job_holders():
    """Test clean_job_description function when job holders are not mentioned in the description."""
    job_title = "Engineer"
    job_description = "This job involves designing and building structures."
    expected_result = "This job involves designing and building structures."
    assert clean_job_description(job_title, job_description) == expected_result


def test_clean_job_description_empty_description():
    """Test clean_job_description function with an empty job description."""
    job_title = "Teacher"
    job_description = ""
    expected_result = ""
    assert clean_job_description(job_title, job_description) == expected_result


def test_clean_job_description_multiple_descriptions():
    """Test clean_job_description function with multiple job descriptions."""
    job_title = "Doctor"
    job_description = "Job holders in this group perform medical tasks."
    expected_result = "Doctor"
    assert clean_job_description(job_title, job_description) == expected_result

    job_description = "This job involves diagnosing and treating patients."
    expected_result = "This job involves diagnosing and treating patients."
    assert clean_job_description(job_title, job_description) == expected_result
