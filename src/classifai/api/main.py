"""Tools to add API functionality."""

from fastapi import FastAPI
import uvicorn
import pandas as pd
import toml

config = toml.load("config.toml")

task = "sic-soc"
endpoint = config["sic-soc"]["endpoint"]
fields = config["all"]["fields"] + config["sic-soc"]["fields"]

# illustrative example of input
data = {"uid": ["0001", "0002"],
        "label": ["1", "5"],
        "confidence_score": [0.79, 0.93],
        "job_title": ["Data Scientist", "Teacher"],
        "job_description": [None, None],
        "level_of_education": [None, None],
        "manage_others": [None, None],
        "industry_descr": [None, None],
}

df = pd.DataFrame(data=data)
df = df.set_index("uid")
result = df.to_json(orient="index")

app = FastAPI()

@app.get(f"/{task}")
def output_json():
    """Deliver classification results."""
    return result

if __name__ == "__main__":
  uvicorn.run("main:app", port=8000, reload=True)