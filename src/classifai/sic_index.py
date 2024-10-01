"""File to process SIC index."""

import pandas as pd
import requests

if __name__ == "__main__":
    # Download extended SIC index
    r = requests.get(
        "https://www.ons.gov.uk/file?uri=/methodology/classificationsandstandards/ukstandardindustrialclassificationofeconomicactivities/uksic2007/uksic2007indexeswithaddendumdecember2022newformatsept20241.xlsx"
    )
    with open(
        "data/sic-index/uksic2007indexeswithaddendumdecember2022newformatsept20241.xlsx",
        "wb",
    ) as outfile:
        outfile.write(r.content)

    # Process extended SIC index
    sic_index_extended = pd.read_excel(
        "data/sic-index/uksic2007indexeswithaddendumdecember2022newformatsept20241.xlsx",
        sheet_name="Alphabetical Index",
        header=2,
        usecols=["UK SIC 2007", "Activity"],
    )

    sic_index_extended.columns = ["sic_code", "description"]

    sic_index_extended.to_csv(
        "data/sic-index/sic_5_digit_extended.csv", index=False
    )

    # Download SIC index
    r = requests.get(
        "    https://www.ons.gov.uk/file?uri=/methodology/classificationsandstandards/ukstandardindustrialclassificationofeconomicactivities/uksic2007/publisheduksicsummaryofstructureworksheet.xlsx"
    )
    with open(
        "data/sic-index/publisheduksicsummaryofstructureworksheet.xlsx",
        "wb",
    ) as outfile:
        outfile.write(r.content)

    # Process 5-digit SIC index
    sic_index = pd.read_excel(
        "data/sic-index/publisheduksicsummaryofstructureworksheet.xlsx",
        sheet_name="reworked structure",
        usecols=["Most disaggregated level", "Description"],
    )

    sic_index.columns = ["description", "sic_code"]

    filtered_sic_index = sic_index[sic_index["sic_code"].str.len() == 5]

    filtered_sic_index.to_csv("data/sic-index/sic_5_digit.csv", index=False)
