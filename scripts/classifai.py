"""Script to instantiate API server and handle tasks."""

# import argparse
from fastapi import FastAPI
from classifai.api import Outputs
import uvicorn
import pandas as pd

endpoint = "classifai"
task = "sic-soc"

# illustrative example of input & dataframe
data = {"uid": ["0001", "0002"],
        "label": ["1", "5"],
        "confidence_score": [0.79, 0.93],
        "job_title": ["Data Scientist", "Teacher"],
        "job_description": [None, None],
        "level_of_education": [None, None],
        "manage_others": [None, None],
        "industry_descr": [None, None],
        "miscellaneous": ["not required", "not required"]
}
df = pd.DataFrame(data=data)

# instantiate FastAPI server instance
app = FastAPI()
tool = Outputs()

@app.get(f"/{endpoint}")
def pipe():

    df = tool.simplify_output(df=df, task=task)
    output = tool.jsonify_output(df=df)
    
    return output
   
if __name__=="__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "-t",
    #     "--task",
    #     type=str,
    #     required=True,
    #     help="name of classification task e.g. 'sic-soc'.",
    # )
    
    # args = vars(parser.parse_args())
    
    # task = args["task"]
    
    uvicorn.run("main:app", port=8000, reload=True)
    
    
    
    