"""Tools to add API functionality."""

import pandas as pd
import toml

config = toml.load("config.toml")


class Outputs():

    def simplify_output(df:pd.DataFrame, task: str = "sic-soc") -> pd.DataFrame:
        fields = config["all"]["fields"] + config[task]["fields"]
        df = df[fields]
        
        return df


    def jsonify_output(df:pd.DataFrame) -> dict:
        df = df.set_index("uid")
        result = df.to_json(orient="index")
        
        return result
  
