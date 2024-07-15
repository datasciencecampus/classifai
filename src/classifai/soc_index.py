"""Download and process the extended SOC framework."""

import numpy as np
import pandas as pd
import requests


class SOCCode:
    """Class to process the extended SOC table."""

    @staticmethod
    def process_soc_code(soc_code: str) -> float:
        """Process 6-digit SOC code.

        Parameters
        ----------
        soc_code : str
            The SOC code to process.

        Returns
        -------
        float
            The processed SOC code. Returns NaN if the SOC code cannot be processed.
        """
        try:
            return int(str(soc_code).replace("/", ""))
        except ValueError:
            return np.NaN

    @staticmethod
    def create_education_table(soc_df: pd.DataFrame) -> pd.DataFrame:
        """Create a table of entry/education requirements for each 4-digit SOC code.

        Parameters
        ----------
        soc_df : pd.DataFrame
            Extended SOC descriptions table with cleaned column names.

        Returns
        -------
        education_descriptions: pd.DataFrame
            Table of entry requirements for each 4-digit SOC code.
        """

        education_descriptions = soc_df.copy()
        education_descriptions = education_descriptions.infer_objects().ffill()
        education_descriptions = education_descriptions[
            education_descriptions["sub_unit_group"]
            == "TYPICAL ENTRY ROUTES AND ASSOCIATED QUALIFICATIONS"
        ]
        education_descriptions = education_descriptions[
            ["unit_group", "descriptions"]
        ]

        education_descriptions = education_descriptions.rename(
            {"unit_group": "join_column"}, axis=1
        )
        education_descriptions = education_descriptions.astype(
            {"join_column": int}
        )

        return education_descriptions

    @staticmethod
    def clean_soc_table(soc_df: pd.DataFrame) -> pd.DataFrame:
        """Clean the SOC and flatten the 6-digit SOC table.

        Parameters
        ----------
        soc_df : pd.DataFrame
            The `Extended SOC descriptions` table with cleaned column names.

        Returns
        -------
        soc_df_clean : pd.DataFrame
            The cleaned SOC dataframe.
        """
        soc_df_clean = soc_df.copy()

        # Fill in the
        for i in range(len(soc_df_clean)):
            for column in [
                "major_group",
                "sub_major_group",
                "minor_group",
                "unit_group",
            ]:
                if isinstance(soc_df_clean.loc[i, column], int):
                    soc_df_clean.at[i, column] = soc_df_clean.loc[
                        i, "group_title"
                    ]
        soc_df_clean[
            ["major_group", "sub_major_group", "minor_group", "unit_group"]
        ] = (
            soc_df_clean[
                ["major_group", "sub_major_group", "minor_group", "unit_group"]
            ]
            .infer_objects()
            .ffill()
        )

        soc_df_clean["soc_code"] = soc_df_clean["sub_unit_group"].apply(
            lambda x: SOCCode.process_soc_code(x)
        )
        soc_df_clean = soc_df_clean.dropna(subset="soc_code")
        soc_df_clean = soc_df_clean.astype({"soc_code": int})
        return soc_df_clean

    @staticmethod
    def join_soc_education_table(
        soc_df: pd.DataFrame, education_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Join the SOC dataframe to the education dataframe.

        Parameters
        ----------
        soc_df : pd.DataFrame
            SOC dataframe created by clean_soc_table.
        education_df : pd.DataFrame
            Education dataframe created by create_education_table.

        Returns
        -------
        pd.DataFrame
            Dataframe with SOC codes, descriptions and entry requirements.
        """
        assert len(education_df.groupby("join_column")) == len(
            education_df
        ), "Duplicate SOC codes in `education_df`"
        soc_df["join_column"] = soc_df["soc_code"].apply(
            lambda x: int(str(x)[0:4])
        )
        overall_df = pd.merge(
            soc_df,
            education_df,
            on="join_column",
            suffixes=["_job", "_education"],
            how="left",
        )

        return overall_df[
            [
                "major_group",
                "sub_major_group",
                "minor_group",
                "unit_group",
                "group_title",
                "descriptions_job",
                "descriptions_education",
                "soc_code",
            ]
        ]


if __name__ == "__main__":
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
    overall_df.to_csv("data/soc-index/soc_6_digit.csv", index=False)
