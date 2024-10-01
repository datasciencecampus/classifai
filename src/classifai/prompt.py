"""The prompt for the SOC classifier."""

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts.prompt import PromptTemplate

from classifai.data_models.response_model import SicSocResponse

# Core prompt

_core_prompt = """You are a conscientious classification assistant of respondent data
for the use in the UK official statistics. Respondent data may be in English or Welsh,
but you always respond in British English."""

# SOC template

_soc_template = """"Given the respondent data (that may include all or some of
job title, job description, level of education, line management responsibilities,
and company's main activity) your task is to determine
the UK SOC (Standard Occupational Classification) code for this job if it can be
determined. Make sure to use the provided 2020 SOC index.

===Respondent Data===
- Job Title: {job_title}
- Job Description: {job_description}
- Company name: {company_name}

===Output Format===
{format_instructions}

===2020 SOC Index===
{soc_index}
"""

parser = PydanticOutputParser(pydantic_object=SicSocResponse)

SOC_PROMPT_PYDANTIC = PromptTemplate.from_template(
    template=_core_prompt + _soc_template,
    partial_variables={
        "format_instructions": parser.get_format_instructions(),
    },
)

# SIC template

_sic_template = """"Given the respondent's description of the main activity their
company does, their job title and job description, your task is to determine
the UK SIC (Standard Industry Classification) code for this company.
Make sure to use the provided 2007 SIC Index.

===Respondent Data===
- Company's main activity: {industry_descr}
- Job Title: {job_title}
- Job Description: {job_description}

===Output Format===
{format_instructions}

===2007 SIC Index===
{sic_index}
"""


parser = PydanticOutputParser(pydantic_object=SicSocResponse)

SIC_PROMPT_PYDANTIC = PromptTemplate.from_template(
    template=_core_prompt + _sic_template,
    partial_variables={
        "format_instructions": parser.get_format_instructions(),
    },
)
