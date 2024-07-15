"""Notebook covering process of generating synthetic survey data."""

# %%
import json

from langchain_community.llms import Ollama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

# number of JSON entries to generate
number = 10

llm = Ollama(model="mistral")
parser = JsonOutputParser()
prompt = """
Generate an example JSON for {number} different employees in an employment survey \
    with the following keys: 'job title', \
    'job description', 'responsibilities', 'manage others', \
    'level of education', 'company', 'industry description'."""
prompt = PromptTemplate(template=prompt, input_variables=["number"])
chain = prompt | llm | parser
synth = chain.invoke({"number": number})

# prettified JSON to file
with open("data/synth_sample.json", "w") as f:
    json.dump(synth, f, indent=2)
