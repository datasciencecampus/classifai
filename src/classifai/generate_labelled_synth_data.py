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


def sample_data(fp: str = "data/SOC2020_coding_index_22-02-24.csv"):
    """Load, slice and sample SOC reference data."""
    df = pd.read_csv(
        "data/SOC2020_coding_index_22-02-24.csv",
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


# %%
# batch_size = 2
# num_records = 4

# llm = Ollama(model="mistral", temperature=0)
# parser = JsonOutputParser()
# prompt = """
# Generate a JSON for SOC code {code} and SOC title {title} \
# with the following keys: \
# 'job_title': str,
# 'job_description': str,
# 'responsibilities': list[str],
# 'manage_others': bool,
# 'level_of_education': str,
# 'company_name': str,
# 'industry_description': str,
# 'SOC_code': int,
# 'SOC_unit_group_title': str
# .
# """
# # prompt = """
# # Generate a list of JSON including one record for each of the {num_rows} SOC code \
# # and SOC title pairs in this dataframe {dataframe} with the following keys: \
# # 'job_title': ,
# # 'job_description': ,
# # 'responsibilities': ,
# # 'manage_others': ,
# # 'level_of_education': ,
# # 'company_name': ,
# # 'industry_description': ,
# # 'SOC_code': ,
# # 'SOC_unit_group_title'.
# # """

# prompt = PromptTemplate(
#     template=prompt, input_variables=["num_rows", "dataframe"]
# )
# chain = prompt | llm | parser

# # df = pl.read_csv(
# #     # https://www.ons.gov.uk/methodology/classificationsandstandards/standardoccupationalclassificationsoc/soc2020/soc2020volume2codingrulesandconventions
# #     "data/SOC2020_coding_index_22-02-24.csv",
# #     encoding="cp1252",
# #     ignore_errors=True
# # ).sample(num_records, seed=42)

# # df = df.select(["SOC 2020", "SOC2020 unit group title"])

# df = pd.read_csv(
#     "data/SOC2020_coding_index_22-02-24.csv",
#     encoding="cp1252",
#     usecols=["SOC 2020", "SOC2020 unit group title"],
# )
# df = df.drop_duplicates().sample(num_records, replace=False, random_state=42)

# synth = []
# counter = 0
# # max_count = int(num_records/batch_size)
# # for i in range(0, num_records-1,batch_size):
# #     counter+=1
# #     print(f"Processing batch {counter} of {max_count}")
# #     batch = df.iloc[i:batch_size+i, ]
# #     # batch = df[i:batch_size+i]
# #     synth.append(chain.invoke({"num_rows": num2words(batch_size), "dataframe": batch}))

# df_list = df.values.tolist()

# for soc in df_list:
#     counter+=1
#     print(f"Processing record {counter} of {num_records}")
#     code = soc[0]
#     title = soc[1]
#     synth.append(chain.invoke({"code": code, "title": title}))

# # # prettified JSON to file
# with open("data/synth_sample_labelled_batched.json", "w") as f:
#     json.dump(synth, f, indent=2)
