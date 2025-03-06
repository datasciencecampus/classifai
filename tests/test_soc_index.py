"""Tests for the SOC index creation."""

import numpy as np
import pandas as pd
import pytest
import requests

from classifai.soc_index import SOCCode


@pytest.mark.skip(reason="Integration test.")
def test_soc_index_length():
    """Integration test to ensure the 6-digit SOC code table is the same length as specified on the ONS website."""

    r = requests.get(
        "https://www.ons.gov.uk/file?uri=/methodology/classificationsandstandards/standardoccupationalclassificationsoc/standardoccupationalclassificationsocextensionproject/extendedsoc2020structureanddescriptionsexcel270524.xlsx"
    )
    with open(
        "data/soc-index/extendedsoc2020structureanddescriptionsexcel270524.xlsx",
        "wb",
    ) as outfile:
        outfile.write(r.content)

    df = pd.read_excel(
        "data/soc-index/extendedsoc2020structureanddescriptionsexcel270524.xlsx",
        sheet_name="Extended SOC descriptions",
        header=1,
        usecols="B:H",
    )

    df.columns = [
        c.strip().lower().replace("-", "_").replace(" ", "_")
        for c in df.columns.values.tolist()
    ]

    # Correcting error in the source data
    df.at[668, "unit_group"] = 2419

    description_df = SOCCode.clean_soc_table(df)
    education_df = SOCCode.create_education_table(df)
    overall_df = SOCCode.join_soc_education_table(description_df, education_df)

    assert len(overall_df) == 1369


@pytest.mark.parametrize(
    ("input_value", "expected_value"),
    [
        ("4547/282722", 4547282722),
        ("Elected officers and representatives", np.nan),
        ("262612871a", np.nan),
    ],
)
def test_process_soc_code(input_value, expected_value):
    """Test SOC codes are processed correctly or nan is returned."""

    output_value = SOCCode.process_soc_code(input_value)
    assert (output_value == expected_value) or (output_value is expected_value)


def test_create_education_table(soc_df_input, soc_df_education):
    """Test the education table is created correctly."""

    output_df = SOCCode.create_education_table(soc_df_input).reset_index(
        drop=True
    )
    pd.testing.assert_frame_equal(
        output_df, soc_df_education, check_dtype=False
    )


def test_clean_soc_table(soc_df_input, soc_df_clean):
    """Test that the SOC table is flattened correctly."""
    output_df = SOCCode.clean_soc_table(soc_df_input)
    pd.testing.assert_frame_equal(output_df, soc_df_clean, check_dtype=False)


def test_join_soc_education_table(soc_df_clean, soc_df_education):
    """Test that the clean SOC table and SOC education table are joined correctly."""

    output_df = pd.DataFrame(
        dict(
            major_group=["Skilled Trades", "Skilled Trades", "Skilled Trades"],
            sub_major_group=[np.nan, "Agriculture", "Agriculture"],
            minor_group=[np.nan, np.nan, np.nan],
            unit_group=[np.nan, "Agriculture", "Agriculture & Related"],
            group_title=[
                "Skilled Trades",
                "Agriculture",
                "Agriculture & Related",
            ],
            descriptions_job=[
                "Skiled Trades work...",
                "Agriculture is...",
                "Agriculture & Related is...",
            ],
            descriptions_education=[np.nan, "GCSEs", np.nan],
            soc_code=[6615, 152523, 626322],
        )
    )

    assert output_df.equals(
        SOCCode.join_soc_education_table(soc_df_clean, soc_df_education)
    )
