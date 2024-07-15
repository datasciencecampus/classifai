"""Notebook: generate synthetic survey data with SOC labels."""

# %%
import json

import pandas as pd
import toml
from langchain_community.llms import Ollama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from tqdm import tqdm

config = toml.load("config.toml")
num_records = config["all"]["num_records"]


def _instantiate_llm(model="mistral", temperature=0):
    llm = Ollama(model=model, temperature=temperature)

    return llm


def _instantiate_parser():
    parser = JsonOutputParser()

    return parser


def _instantiate_prompt():
    prompt_text = config["soc"]["prompt"]
    prompt = PromptTemplate(
        template=prompt_text, input_variables=["num_rows", "dataframe"]
    )

    return prompt


def llm_chain():
    """Instantiate LLM chain of prompt, llm and output parser."""
    prompt = _instantiate_prompt()
    llm = _instantiate_llm()
    parser = _instantiate_parser()
    chain = prompt | llm | parser

    return chain


# https://www.ons.gov.uk/methodology/classificationsandstandards/standardoccupationalclassificationsoc/soc2020/soc2020volume2codingrulesandconventions
def sample_data(fp: str = "data/SOC2020_coding_index_22-02-24.csv"):
    """Load, slice and sample SOC reference data."""
    df = pd.read_csv(
        fp,
        encoding="cp1252",
        usecols=["SOC 2020", "SOC2020 unit group title"],
    )
    df = df.drop_duplicates().sample(
        num_records, replace=False, random_state=42
    )

    return df


def save_json(output: list[dict]):
    """Store synthetic survey data to JSON."""
    with open("data/synth_sample_labelled_batched.json", "w") as f:
        json.dump(output, f, indent=2)

    return None


def main():
    """Script to generate and store synthetic survey data."""
    chain = llm_chain()
    data = sample_data()

    synth = []
    counter = 0

    df_list = data.values.tolist()

    for soc in tqdm(df_list, desc="Records processing progress..."):
        counter += 1
        # print(f"Processing record {counter} of {num_records}")
        code = soc[0]
        title = soc[1]
        synth.append(chain.invoke({"code": code, "title": title}))

    save_json(synth)


if __name__ == "__main__":
    main()
