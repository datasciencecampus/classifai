"""Notebook: generate synthetic survey data with SOC labels."""

# %%
import json

import polars as pl
from langchain_community.llms import Ollama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

# Number of records to generate
num_samples = 10

df = pl.read_csv(
    # https://www.ons.gov.uk/methodology/classificationsandstandards/standardoccupationalclassificationsoc/soc2020/soc2020volume2codingrulesandconventions
    "data/SOC2020_coding_index_22-02-24.csv",
    encoding="cp1252",
    ignore_errors=True,
)
df = df.select(["SOC 2020", "SOC2020 unit group title"])

sample = df.sample(num_samples)

llm = Ollama(model="mistral", temperature=0)
parser = JsonOutputParser()
prompt = """

Generate an example JSON including one employee for each of the ten SOC code \
and SOC title pairs in this dataframe {dataframe} with the following keys: \
'job_title': ,
'job_description': ,
'responsibilities': ,
'manage_others': ,
'level_of_education': ,
'company': ,
'industry_description': ,
'SOC_code': ,
'SOC_unit_group_title'.
"""

prompt = PromptTemplate(
    template=prompt, input_variables=["soc_code", "soc_title"]
)
chain = prompt | llm | parser

synth = chain.invoke({"dataframe": sample})

# # prettified JSON to file
with open("data/synth_sample_labelled.json", "w") as f:
    json.dump(synth, f, indent=2)
