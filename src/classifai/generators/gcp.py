import json

import pandas as pd
from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions

from .base import GeneratorBase

#
#
#
########################
#### SYSTEM PROMPTS FOR DIFFERENT TASK TYPES: currently, Classification or Summarization.
########################


SUMMARISE_SYSTEM_PROMPT = """You are an advanced AI assistant designed to provide accurate and helpful responses. You have access to a knowledge base that retrieves relevant information based on user queries. The retrieved information will be provided to you as context in an XML format before the user query. Use this context to generate your response. If the context does not fully address the query, use your general knowledge to provide a complete and accurate answer.

Guidelines:
1. Always prioritize the provided context when generating your response.
2. The context will be provided as an XML structure containing multiple entries. Each entry represents a relevant piece of information retrieved from the knowledge base.
3. Use the retrieved entries to enhance your response. Summarize or synthesize the information as needed to address the user query.
4. Be concise, clear, and factual in your answers.
5. Avoid making up information. If you are unsure, state that you do not have enough information to answer the query.

The XML structure for the context and user query will be as follows:
<Context>
    <Entry>
        <ID>[ID of the first entry]</ID>
        <Text>[Text from the first retrieved entry]</Text>
    </Entry>
    <Entry>
        <ID>[ID of the second entry]</ID>
        <Text>[Text from the second retrieved entry]</Text>
    </Entry>
    ...
</Context>

<UserQuery>
    <Text>[The user’s query will be inserted here]</Text>
</UserQuery>

Your task is to generate a response based on the context and the user query."""


RERANK_SYSTEM_PROMPT = """"""

RERANK_RESPONSE_SCHEMA = {}


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
    <Text>[The user’s query will be inserted here]</Text>
</UserQuery>

Your task is to analyze the context and the user query, and return the classification in the required structured format."""

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


#
#
#
########################
#### GENERAL FUNCTION FOR FORMATTING THE USER QUERY PROMPT WITH RETRIEVED RESULTS RELIES ON VECTOR STORE RESULTS DATAFRAME
########################


def format_prompt_with_retrieval_results(user_query: str, df: pd.DataFrame) -> str:
    """
    Generates a formatted XML prompt for the generative model.

    Args:
        user_query (str): The user's query.
        df (pd.DataFrame): A DataFrame containing 'doc_text' and 'doc_id' columns.

    Returns:
        str: The formatted XML prompt.å
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
    formatted_prompt = f"""<Context>
{context_entries}
</Context>

<UserQuery>
    <Text>{user_query}</Text>
</UserQuery>"""

    return formatted_prompt


#
#
#
########################
#### FORMATTING FUNCTIONS THAT INTERPRET THE MODEL RAW RESPONSE AND FORMAT TO OUR STANDARD OUTPUT OR THROW ERROR
########################


def format_summary_output(generated_text, ranking):
    return {"text": generated_text, "ranking": ranking}


def format_reranker_output(generated_text, ranking):
    # TODO: here we want to add something that will parse the re-ranking from the generated text and apply it to the ranking list
    return {"text": generated_text, "ranking": ranking}


def format_classification_ouput(generated_text, ranking):
    ### could do additional package cheks to validate that response is well generated and throw error if not

    generated_text = json.loads(generated_text)
    return {"text": f"classification: {generated_text['classification']}", "ranking": ranking}


#
#
#
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

        # decide logic for summarization and reranking
        if task_type == "summarization":
            self.system_prompt = SUMMARISE_SYSTEM_PROMPT
            self.response_formatting_function = format_summary_output
            self.response_schema = None
        elif task_type == "reranking":
            self.system_prompt = RERANK_SYSTEM_PROMPT
            self.response_formatting_function = format_reranker_output
            self.response_schema = None
        elif task_type == "classification":
            self.system_prompt = CLASSIFICATION_SYSTEM_PROMPT
            self.response_formatting_function = format_classification_ouput
            self.response_schema = CLASSIFICATION_RESPONSE_SCHEMA

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
                response_mime_type="application/json" if self.response_schema else "text/plain",
                response_schema=self.response_schema,
            ),
        )

        # Process and return the final output to get text and ranking response
        formatted_response = self.response_formatting_function(response.text, vectorstore_search_results)

        return formatted_response
