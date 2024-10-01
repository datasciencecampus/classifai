"""Class for the LLM."""

from functools import lru_cache

from langchain.chains.llm import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain_google_vertexai import VertexAI

from classifai.data_models.response_model import SicSocResponse
from classifai.prompt import SIC_PROMPT_PYDANTIC, SOC_PROMPT_PYDANTIC


class ClassificationLLM:
    """Class for LLM.

    Wraps the logic for using an LLM to classify respondent's data
    based on provided index.

    Args:
        model_name (str): Name of the model. Defaults to `gemini-pro`.
        max_tokens (int): Maximum number of tokens to generate. Defaults to 1600.
        temperature (float): Temperature of the LLM model. Defaults to 0.0.
    """

    def __init__(
        self,
        model_name: str = "gemini-1.0-pro",
        max_tokens: int = 1600,
        temperature: float = 0.0,
    ):
        """Initialise the ClassificationLLM object."""

        self.llm = VertexAI(
            model_name=model_name,
            max_output_tokens=max_tokens,
            temperature=temperature,
            location="europe-west2",
        )

        self.soc_prompt = SOC_PROMPT_PYDANTIC
        self.sic_prompt = SIC_PROMPT_PYDANTIC

    @lru_cache
    def get_soc_code(
        self,
        job_title: str,
        job_description: str,
        company_name: str,
        soc_index: str,
    ) -> SicSocResponse:
        """Generate a SOC classification based on respondent's data.

        Uses a whole condensed index embedded in the query.

        Args:
            job_title (str): The title of the job.
            job_description (str): The description of the job.
            company_name (str): The name of the company.
            soc_index (str): The SOC index separated by newline characters.

        Returns
        -------
            SicSocResponse: The generated response to the query.

        """
        chain = LLMChain(llm=self.llm, prompt=self.soc_prompt)
        response = chain.invoke(
            {
                "job_title": job_title,
                "job_description": job_description,
                "company_name": company_name,
                "soc_index": soc_index,
            },
            return_only_outputs=True,
        )
        # Parse the output to desired format with one retry
        parser = PydanticOutputParser(pydantic_object=SicSocResponse)

        validated_answer = parser.parse(response["text"])

        return validated_answer

    @lru_cache
    def get_sic_code(
        self,
        industry_descr: str,
        job_title: str,
        job_description: str,
        sic_index: str,
    ) -> SicSocResponse:
        """
        Generate a SOC classification based on respondent's data.

        Use a whole condensed index embedded in the query.

        Args:
            industry_descr (str): Description of the industry.
            job_title (str): Title of the job.
            job_description (str): Description of the job.
            sic_index (str): The SIC index separated by newline characters.

        Returns
        -------
            SicSocResponse: Generated response to the query.
        """

        chain = LLMChain(llm=self.llm, prompt=self.sic_prompt)
        response = chain.invoke(
            {
                "industry_descr": industry_descr,
                "job_title": job_title,
                "job_description": job_description,
                "sic_index": sic_index,
            },
            return_only_outputs=True,
        )
        # Parse the output to desired format with one retry
        parser = PydanticOutputParser(pydantic_object=SicSocResponse)

        validated_answer = parser.parse(response["text"])

        return validated_answer
