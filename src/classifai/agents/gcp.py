import json

import pandas as pd
from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions
from pydantic import BaseModel, Field

from classifai.indexers.dataclasses import VectorStoreSearchOutput

from .base import GeneratorBase

########################
#### SYSTEM PROMPTS FOR DIFFERENT TASK TYPES: currently, Classification or Summarization.
########################

CLASSIFICATION_SYSTEM_PROMPT = """You are an AI assistant designed to classify a user query based on the provided context. You will be provided with 5 candidate entries retrieved from a knowledge base, each containing an ID and a text description. Your task is to analyze the user query and the text of the context entries to determine which of the entries best matches the user query.

Guidelines:
1. Always prioritize the provided context when making your classification.
2. The context will be provided as an XML structure containing multiple entries. Each entry includes an ID and a text description.
3. The IDs will be integer values from 0 to 4, corresponding to the 5 candidate entries.
4. Use the text of the entries to determine the most relevant classification for the user query.
5. Your output must be a JSON object that adheres to the following schema:
    - The JSON object must contain a single key, `classification`.
    - The value of `classification` must be an integer between 0 and 4, representing the ID of the best matching entry.
    - If no classification can be determined due to ambiguity or insufficient information, the value of `classification` must be `-1`.

Example of the required JSON output:
{
     "classification": 1
}

The XML structure for the context and user query will be as follows:
<Context>
     <Entry>
          <ID>0</ID>
          <Text>[Text from the first entry]</Text>
     </Entry>
     <Entry>
          <ID>1</ID>
          <Text>[Text from the second entry]</Text>
     </Entry>
     ...
     <Entry>
          <ID>4</ID>
          <Text>[Text from the fifth entry]</Text>
     </Entry>
</Context>

<UserQuery>
     <Text>[The user query will be inserted here]</Text>
</UserQuery>

Your task is to analyze the context and the user query, and return the classification in the required structured format."""


########################
#### GENERAL FUNCTION FOR FORMATTING THE USER QUERY PROMPT WITH RETRIEVED RESULTS FROM VECTORSTORE
########################


def format_prompt_with_retrieval_results(df: pd.DataFrame) -> str:
    """Generates a formatted XML prompt for the generative model from a structured DataFrame.

    Args:
        df (pd.DataFrame): A DataFrame containing columns as per `searchOutputSchema`.

    Returns:
        str: The formatted XML prompt.
    """
    # Extract the user query (assuming all rows have the same query_id and query_text)
    user_query = df["query_text"].iloc[0]

    # Limit to the top 5 entries based on rank
    top_entries = df.nsmallest(5, "rank")

    # Build the <Context> section
    context_entries = "\n".join(
        f"    <Entry>\n        <ID>{idx}</ID>\n        <Text>{row['doc_text']}</Text>\n    </Entry>"
        for idx, row in top_entries.iterrows()
    )

    # Combine everything into the final prompt
    formatted_prompt = f"""
<Context>
{context_entries}
</Context>

<UserQuery>
    <Text>{user_query}</Text>
</UserQuery>"""

    return formatted_prompt


########################
#### SYSTEM PROMPTS FOR DIFFERENT TASK TYPES: Classification
########################


class ClassificationResponseModel(BaseModel):
    classification: int = Field(description="Chosen ID of the best matching entry.", ge=-1)


########################
#### FORMATTING FUNCTIONS THAT INTERPRET THE MODEL RAW RESPONSE, FORMATS, and APPLIES TO DF
########################


def format_classification_output(generated_text, result: VectorStoreSearchOutput) -> VectorStoreSearchOutput:
    # Parse the generated text
    try:
        response = json.loads(generated_text)
        validated_response = ClassificationResponseModel(**response)
    except (json.JSONDecodeError, ValueError):
        # If parsing or validation fails, return the original DataFrame
        return result

    # Extract the classification
    classification = validated_response.classification

    # Validate the classification value is in the expected range
    MIN_INDEX = 0
    MAX_INDEX = 4
    if int(classification) < MIN_INDEX or int(classification) > MAX_INDEX:
        return result

    # Otherwise, filter to only keep the row with the classified doc_id
    result = result.iloc[[classification]].reset_index(drop=True)

    return VectorStoreSearchOutput(result)


########################
#### ACTUAL AGENT CODE
########################


class GcpAgent(GeneratorBase):
    def __init__(
        self,
        project_id: str,
        location: str,
        model_name: str = "gemini-3-flash-preview",
        task_type: str = "classification",
    ):
        self.client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location,
            http_options=HttpOptions(api_version="v1"),
        )

        # assign model name and vectorstore isntance
        self.model_name = model_name

        # decide logic for classification or reranking
        # if task_type == "reranking":
        #     self.system_prompt = RERANK_SYSTEM_PROMPT
        #     self.response_formatting_function = format_reranking_output
        #     self.response_schema = RERANK_RESPONSE_SCHEMA
        if task_type == "classification":
            self.system_prompt = CLASSIFICATION_SYSTEM_PROMPT
            self.response_formatting_function = format_classification_output
            self.response_schema = ClassificationResponseModel.model_json_schema()

        else:
            raise ValueError(
                f"Unsupported task_type: {task_type}. Current supported types are 'reranking' and 'classification'."
            )

    def transform(self, results: VectorStoreSearchOutput) -> VectorStoreSearchOutput:
        # Group rows by query_id and process individually
        grouped = list(results.groupby("query_id"))
        all_results = []

        # Iterate over each group (query_id)
        for _, group in grouped:
            # Create a prompt for the current query_id
            prompt = format_prompt_with_retrieval_results(group)

            # Prompt the model with the single prompt
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=GenerateContentConfig(
                    system_instruction=self.system_prompt,
                    response_mime_type="application/json",
                    response_schema=self.response_schema,
                ),
            )

            # Process the response from the genai
            formatted_result = self.response_formatting_function(response.text, group)
            all_results.append(formatted_result)

        # Combine all results into the final DataFrame
        final_results = pd.concat(all_results, ignore_index=True)

        return VectorStoreSearchOutput(final_results)
