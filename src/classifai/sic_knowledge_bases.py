"""Script to prepare SIC knowledge bases provided in Excel for embedding."""

import pandas as pd

from src.classifai.utils import get_secret

if __name__ == "__main__":
    google_api_key = get_secret("GOOGLE_API_KEY")
    app_data_bucket = get_secret("APP_DATA_BUCKET")

    master_sic = pd.read_excel(
        f"gs://{app_data_bucket}/ACTR Main.xlsx",
        sheet_name="Master",
        usecols=["SIC07", "BRIDGE", "DESCRIPTION"],
    )

    master_sic = master_sic.rename(
        columns={
            "SIC07": "label",
            "BRIDGE": "bridge",
            "DESCRIPTION": "description",
        }
    )
    master_sic = master_sic.dropna(subset="description")

    master_sic["description"] = (
        master_sic["description"].str.lower().str.capitalize()
    )
    master_sic["description"] = master_sic["description"].str.strip()

    master_sic = master_sic[["description", "label", "bridge"]]

    master_sic.to_csv("data/sic_index/sic_knowledge_base.csv")
