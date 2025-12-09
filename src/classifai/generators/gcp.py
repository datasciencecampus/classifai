import json

import pandas as pd
from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions
from pydantic import BaseModel, Field

from .base import GeneratorBase

########################
#### SYSTEM PROMPTS FOR DIFFERENT TASK TYPES: currently, Classification or Summarization.
########################

CLASSIFICATION_SYSTEM_PROMPT = """You are an advanced AI assistant designed to classify a user query based on the provided context. The context consists of multiple entries retrieved from a knowledge base, each containing an ID and a text description. Your task is to analyze the user query and the text of the context entries to determine which entry ID best matches the query.

Guidelines:
1. Always prioritize the provided context when making your classification.
2. The context will be provided as an XML structure containing multiple entries. Each entry includes an ID and a text description.
3. Use the text of the entries to determine the most relevant classification for the user query.
4. Your output must be a structured response in the following format:
   {'classification': <entry_id>}
   - Replace <entry_id> with the ID of the entry that best matches the query.
5. If the correct classification cannot be determined due to ambiguity or insufficient information, your output should be:
   {'classification': CBA} CBA stands for "Cannot Be Assigned". It is very important that you generate CBA in this case to indicate that it can't be assigned.
6. Be concise and factual. Do not include any additional information or explanations in your output.

The XML structure for the context and user query will be as follows:
<Context>
    <Entry>
        <ID>[ID of the first entry]</ID>
        <Text>[Text from the first entry]</Text>
    </Entry>
    <Entry>
        <ID>[ID of the second entry]</ID>
        <Text>[Text from the second entry]</Text>
    </Entry>
    ...
</Context>

<UserQuery>
    <Text>[The user query will be inserted here]</Text>
</UserQuery>

Your task is to analyze the context and the user query, and return the classification in the required structured format."""


RERANK_SYSTEM_PROMPT = """
You are an advanced AI assistant designed to re-rank a set of entries from a knowledge base based on their relevance to a user query.
Your task is to analyze the user query and the text of the context entries, then return a ranked list of entry IDs from most to least relevant.

### Guidelines:
1. Use the provided context and user query to determine relevance. Focus on semantic meaning and prioritize entries that directly address the query.
2. The context will be provided as an XML structure containing multiple entries. Each entry includes an ID and a text description.
3. Your output must be a structured response in the following format:
    {"ranking": ["<entry_id_1>", "<entry_id_2>", ...]}
    - Replace `<entry_id_1>`, `<entry_id_2>`, etc., with the IDs of the entries, ordered from most to least relevant to the user query.
4. Ensure that:
    - Each `<entry_id>` appears only once in the ranking.
    - The ranking contains the same number of entries as provided in the context.
5. If you cannot determine a ranking due to ambiguity or insufficient information, your output should be:
    {"ranking": []}
    - An empty list indicates that no ranking can be assigned.
6. Be concise and factual. Do not include any additional information or explanations in your output.

### Context and Query Format:
<Context>
    <Entry>
        <ID>[ID of the first entry]</ID>
        <Text>[Text from the first entry]</Text>
    </Entry>
    <Entry>
        <ID>[ID of the second entry]</ID>
        <Text>[Text from the second entry]</Text>
    </Entry>
    ...
</Context>

<UserQuery>
    <Text>[The user query will be inserted here]</Text>
</UserQuery>

### Your Task:
Analyze the context and the user query, and return the ranking in the required structured format.
"""


########################
#### SYSTEM PROMPTS FOR DIFFERENT TASK TYPES: currently, Classification or Summarization.
########################


CLASSIFICATION_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "classification": {
            "type": "STRING",
            "description": "The ID of the entry that best matches the user query, or 'CBA' if no classification can be determined. MUST be exactly 'CBA' when no classification can be assigned.",
        },
    },
    "required": ["classification"],
    "additionalProperties": False,
}

RERANK_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "ranking": {
            "type": "ARRAY",
            "items": {
                "type": "STRING",
                "description": "The ID of an entry, ordered from most to least relevant to the user query.",
            },
            "description": "A ranked list of entry IDs from most to least relevant. MUST be an empty list if no ranking can be assigned.",
        },
    },
    "required": ["ranking"],
    "additionalProperties": False,
}


########################
#### FORMATTING FUNCTIONS THAT INTERPRET THE MODEL RAW RESPONSE AND FORMAT TO OUR STANDARD OUTPUT OR THROW ERROR
########################


def format_classification_output(generated_text, df: pd.DataFrame) -> pd.DataFrame:
    class ClassificationResponseModel(BaseModel):
        classification: str = Field(
            ...,
            description="The ID of the entry that best matches the user query, or 'CBA' if no classification can be determined.",
        )

    # Parse the generated text
    try:
        response = json.loads(generated_text)
        validated_response = ClassificationResponseModel(**response)
    except json.JSONDecodeError as e:
        raise ValueError("Generative model output is bad.") from e

    # Validate the response against the schema

    # Extract the classification
    classification = validated_response.classification

    # If classification is 'CBA' or not present in the DataFrame, return the whole DataFrame
    if classification == "CBA" or classification not in df["doc_id"].values:
        return df

    # Otherwise, filter to only keep the row with the classified doc_id
    df = df[df["doc_id"] == classification]

    return df


def format_reranking_output(generated_text, df: pd.DataFrame) -> pd.DataFrame:
    class RerankResponseModel(BaseModel):
        ranking: list[str] = Field(
            ...,
            description="A ranked list of entry IDs from most to least relevant. MUST be an empty list if no ranking can be assigned.",
        )

    # Parse the generated text
    try:
        response = json.loads(generated_text)
        validated_response = RerankResponseModel(**response)
    except json.JSONDecodeError as e:
        raise ValueError("Generative model output is bad.") from e

    # Extract the reranking list
    rerank_index = validated_response.ranking

    # Reorder the DataFrame based on rerank_index
    df = df.set_index("doc_id").loc[rerank_index].reset_index()

    # Add a new column 'reranking' with values from 0 to N-1
    df["reranking"] = range(len(df))

    return df


########################
#### GENERAL FUNCTION FOR FORMATTING THE USER QUERY PROMPT WITH RETRIEVED RESULTS RELIES ON VECTOR STORE RESULTS DATAFRAME
########################


def format_prompt_with_retrieval_results(user_query: str, df: pd.DataFrame) -> str:
    """Generates a formatted XML prompt for the generative model.

    Args:
        user_query (str): The user's query.
        df (pd.DataFrame): A DataFrame containing 'doc_text' and 'doc_id' columns.

    Returns:
        str: The formatted XML prompt.
    """
    # Ensure the DataFrame has the required columns
    if not {"doc_text", "doc_id"}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'doc_text' and 'doc_id' columns.")

    # Limit to the top 5 entries
    top_entries = df.head(5)

    # Build the <Context> section
    context_entries = "\n".join(
        f"    <Entry>\n        <ID>{row['doc_id']}</ID>\n        <Text>{row['doc_text']}</Text>\n    </Entry>"
        for _, row in top_entries.iterrows()
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
#### ACTUAL AGENT CODE
########################


class RagAgent_GCP(GeneratorBase):
    def __init__(
        self,
        project_id: str,
        location: str = "europe-west2",
        model_name: str = "gemini-2.5-flash",
        vectorStore=None,
        task_type: str = "summarization",
    ):
        self.client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location,
            http_options=HttpOptions(api_version="v1"),
        )

        # assign model name and vectorstore isntance
        self.model_name = model_name
        self.vectorStore = vectorStore

        # decide logic for classification or reranking
        if task_type == "reranking":
            self.system_prompt = RERANK_SYSTEM_PROMPT
            self.response_formatting_function = format_reranking_output
            self.response_schema = RERANK_RESPONSE_SCHEMA
        elif task_type == "classification":
            self.system_prompt = CLASSIFICATION_SYSTEM_PROMPT
            self.response_formatting_function = format_classification_output
            self.response_schema = CLASSIFICATION_RESPONSE_SCHEMA

        else:
            raise ValueError(
                f"Unsupported task_type: {task_type}. Current supported types are 'reranking' and 'classification'."
            )

    def transform(self, prompt: str):
        # Query the knowledgebase
        vectorstore_search_results = self.vectorStore.search(prompt, n_results=5)

        # Format the retrieved information for the prompt injecting into the prompt
        final_prompt = format_prompt_with_retrieval_results(prompt, vectorstore_search_results)

        # Prompt the model
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=final_prompt,
            config=GenerateContentConfig(
                system_instruction=self.system_prompt,
                response_mime_type="application/json",
                response_schema=self.response_schema,
            ),
        )

        # Process and return the final output to get text and ranking response
        formatted_response = self.response_formatting_function(response.text, vectorstore_search_results)

        return formatted_response
