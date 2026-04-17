import io
import json
from collections.abc import Callable

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from classifai._optional import check_deps
from classifai.exceptions import ConfigurationError, HookError
from classifai.indexers.dataclasses import VectorStoreSearchOutput
from classifai.indexers.hooks.hook_factory import HookBase


class DeduplicationHook(HookBase):
    """A post-processing hook to remove duplicate knowledgebase entries, i.e. entries with the same label."""

    def _mean_score(self, scores):
        return np.mean(scores)

    def _max_score(self, scores):
        return np.max(scores)

    def __init__(self, score_aggregation_method: str = "max"):
        """Inititialises the hook with the specified method for assigning scores to deduplicated entries.

        Args:
            score_aggregation_method (str): Method for assigning score to the deduplicated entry.
                Must be one of "max" or "mean". Defaults to "max".
                A future update will introduce a 'softmax' option.
        """
        if score_aggregation_method not in ["max", "mean"]:
            raise HookError(
                "Invalid method for DeduplicationHook. Must be one of 'max', or 'mean'.",
                context={self.hook_type: "Deduplication", "method": score_aggregation_method},
            )
        self.score_aggregation_method = score_aggregation_method
        if self.score_aggregation_method == "max":
            self.score_aggregator = self._max_score
        elif self.score_aggregation_method == "mean":
            self.score_aggregator = self._mean_score

        super().__init__(hook_type="post_processing")

    def __call__(self, input_data: VectorStoreSearchOutput) -> VectorStoreSearchOutput:
        """Aggregates retrieved knowledgebase entries corresponding to the same label."""
        # 1) Group on two levels - first on query_id, then on doc_label, to ensure that entries with the same label are
        #    deduplicated within the results for each query. Note that there is a 1-1 mapping between query_id and query_text,
        #    so no extra grouping is made, but this excludes query_text from the columns to be processed.
        # 2) For each group, aggregate the score using the specified method, and assign a new column 'idxmax' to the unique id
        #    of the entry with the best score. This will allow us to retain the metadata of the best scoring entry.
        df_gpby = (
            input_data.groupby(["query_id", "query_text", "doc_label"])
            .aggregate(
                score=("score", self.score_aggregator),
                idxmax=("score", "idxmax"),
                rank=("rank", "min"),
            )
            .reset_index()
        )
        # For each query, re-assign ranks based on the new aggregated scores, to the remaining entries, to ensure that the best
        # scoring entry for each label is ranked highest.
        for query in df_gpby["query_id"].unique():
            batch = df_gpby[df_gpby["query_id"] == query]
            new_rank = pd.factorize(-batch["score"], sort=True)[0] + 1
            df_gpby.loc[batch.index, "rank"] = new_rank
        # Finally, we re-merge the deduplicated results with the original input dataframe,
        # to retrieve the metadata of the best scoring entry for each label, and return the processed output.
        for col in set(input_data.columns).difference(set(df_gpby.columns)):
            df_gpby[col] = df_gpby["idxmax"].map(input_data[col])
        # We sort the output by query_id and doc_label to ensure a consistent order of results for each query,
        # and validate the output against the dataclass schema.
        processed_output = input_data.__class__.validate(
            df_gpby[input_data.columns].sort_values(by=["query_id", "doc_label"], ascending=[True, True])
        )
        return processed_output


class RagHook(HookBase):
    """A post-processing hook to perform Retrieval Augmented Generation."""

    def __init__(  # noqa: PLR0913
        self,
        context_prompt: str = "",
        response_template: str = "",
        llm_response_parser: Callable[[VectorStoreSearchOutput, str], str] | None = None,
        project_id: str | None = None,
        api_key: str | None = None,
        location: str = "europe-west2",
        model_name: str = "gemini-2.5-flash",
        **client_kwargs,
    ):
        """Initializes the GcpVectoriser with the specified project ID, location, and model name.

        Args:
            context_prompt (str): [optional] A prompt to provide context to the LLM for RAG. Defaults to "".
            response_template (str): [optional] A template for formatting the response. Defaults to "{}".
            llm_response_parser (Callable[[VectorStoreSearchOutput, str], list]): A callable method for parsing
                the LLM response (str) within the context of the search output for a single query
                (VectorStoreSearchOutput). Defaults to using `_default_parse_LLM_response`, which assumes the
                response to be a valid JSON array of strings.
            project_id (str): [optional] The Google Cloud project ID. Defaults to None.
            api_key (str): [optional] The API key for authenticating with the GenAI API. Defaults to None.
            location (str): [optional] The location of the GenAI API. Defaults to None.
            model_name (str): [optional] The name of the generative model. Defaults to "gemini-2.5-flash".
            **client_kwargs: Additional keyword arguments to pass to the GenAI client,
                             e.g. vertexai=True.

        Raises:
            ConfigurationError: If the GenAI client fails to initialize.
        """
        check_deps(["google-genai"], extra="gcp")
        from google import genai  # type: ignore

        self.model_name = model_name
        self.context_prompt = context_prompt
        self.response_template = response_template
        self.llm_response_parser = llm_response_parser or self._default_parse_LLM_response

        if project_id and not api_key:
            client_kwargs.setdefault("project", project_id)
            client_kwargs.setdefault("location", location)
        elif api_key and not project_id:
            client_kwargs.setdefault("api_key", api_key)
        else:
            raise ConfigurationError(
                "Provide either 'project_id' and 'location' together, or 'api_key' alone.",
                context={"hooks": "RAG"},
            )

        self.client_kwargs = client_kwargs

        try:
            self.client = genai.Client(**self.client_kwargs)  # .aio
            self.config_generator = genai.types.GenerateContentConfig  # type: ignore
        except Exception as e:
            raise ConfigurationError(
                "Failed to initialize GCP GenAI client.",
                context={"hooks": "RAG", "cause": str(e), "cause_type": type(e).__name__},
            ) from e

    def _format_prompt_single_query(self, search_subset: VectorStoreSearchOutput, query_id: str):
        """Format a prompt directing the LLM to process search responses corresponding to a single
        origin query. The prompt includes instructions, search output schema, description for formatting
        the response, and the search output itself.

        Args:
            search_subset (VectorStoreSearchOutput): A subset of the search output corresponding to a single query.
            query_id (str): The ID of the query corresponding to the search subset, used for prompt formatting.

        Returns:
            str: The formatted prompt for the LLM.
        """
        schema_info_buffer = io.StringIO()
        search_subset.head(n=0).info(verbose=True, show_counts=False, memory_usage=False, buf=schema_info_buffer)
        schema_info = schema_info_buffer.getvalue()
        schema_info_buffer.close()
        return f"""
        Instructions:
        -------------
        Process the data provided in the Data section (format described in the Input Format section) according to the
        task description given in your system prompt. Use the Output Format section to format your response.

        Input Format:
        -------------
        The Data will be a Pandas DataFrame, converted to JSON. The schema of the DataFrame is described as follows:
        {schema_info}

        Output Format:
        --------------
        The output, for each row in the Data section, should be formatted as follows:
        {self.response_template}

        Data:
        -----
        {search_subset.to_json()}
        """

    @staticmethod
    def _default_parse_LLM_response(search_subset: VectorStoreSearchOutput, llm_response: str) -> list[str]:
        """Parse the LLM response as JSON, expected to form a list of strings.

        Args:
            search_subset (VectorStoreSearchOutput): The search output corresponding to a single query.
            llm_response (str): The raw text response from the LLM.

        Returns:
            (list[str]): The parsed response, which can be assigned to the `RAG_response` column.

        Raises:
            ValueError: If the LLM response cannot be parsed as JSON.
        """
        try:
            parsed_response = json.loads(llm_response)
            if not isinstance(parsed_response, list):
                raise HookError(
                    "LLM response could not be parsed from JSON as a list", context={"postprocessing": "RAGHook"}
                )
            if len(parsed_response) != len(search_subset):
                raise HookError(
                    "LLM response length does not match search output length", context={"postprocessing": "RAGHook"}
                )
        except json.JSONDecodeError as e:
            raise HookError(
                "The LLM response could not be parsed as valid JSON",
                context={"postprocessing": "RAGHook", "error": f"{e}", "response": llm_response},
            ) from e
        except Exception as e:
            raise HookError(
                "The LLM response parsing failed for an unknown reason",
                context={"postprocessing": "RAGHook", "error": f"{e}"},
            ) from None
        return parsed_response

    def _call_llm(self, search_output: VectorStoreSearchOutput) -> str:
        """Calls the LLM to generate responses for each query in the search output, using the formatted prompts.

        Args:
            search_output (VectorStoreSearchOutput): The output from the `.search()` method.

        Returns:
            (VectorStoreSearchOutput): The search output with an additional column for the LLM-generated RAG response.

        Notes:
            - This method adds a new column `RAG_response` to the VectorStoreSearchOutput object. The format of the response
              is user-specified, via the `response_template` parameter of the hook, and is parsed by the `llm_response_parser`
              parameter of the hook (defaulting to attempting to parse the response as a JSON array of strings if omitted).
            - Each unique query in the search output is processed separately, with a prompt formatted using the
              `_format_prompt_single_query` method.
        """
        updated_search_output = search_output.copy()
        updated_search_output["RAG_response"] = ""
        distinct_queries = search_output["query_id"].unique()
        for query_id in distinct_queries:
            search_subset = search_output[search_output["query_id"] == query_id]
            prompt = self._format_prompt_single_query(search_subset, query_id)
            response = self.client.models.generate_content(  # await ...
                model=self.model_name,
                contents=prompt,
                config=self.config_generator(system_instruction=self.context_prompt),
            )
            updated_search_output.loc[search_subset.index, "RAG_response"] = self.llm_response_parser(
                search_subset, response.text
            )
        return updated_search_output

    def __call__(self, search_output: VectorStoreSearchOutput) -> VectorStoreSearchOutput:
        """Calls the LLM to add the `RAG_response` column."""
        processed_output = self._call_llm(search_output)
        return processed_output


class HuggingFaceRagHook(HookBase):
    """A post-processing hook to perform Retrieval Augmented Generation using Hugging Face Inference API.
    IMPORTANT NOTE: Use of this hook means search output data will be sent to HuggingFace, so ensure that this is
    compliant with your data privacy and security requirements.
    """

    def __init__(  # noqa: PLR0913
        self,
        context_prompt: str,
        response_prompt: str,
        api_key: str,
        llm_response_pydantic_model: BaseModel | Callable | None = None,
        model_name: str = "deepseek-ai/DeepSeek-v3.2",
        model_provider: str | None = None,
        visible_cols: list[str] | None = None,
        **client_kwargs,
    ):
        """Initializes the HuggingFaceRagHook with the specified model name and API key.

        Args:
            context_prompt (str): A prompt to provide context to the LLM for RAG.
            response_prompt (str): A template for formatting the response.
            api_key (str): The API key for authenticating with the Hugging Face Inference API.
            llm_response_pydantic_model (BaseModel | Callable): A pydantic model or a callable function for
                parsing and validating the LLM response, which can be used to enforce a specific response schema.
                The pydantic model must have a list attribute, `answer`, which will be assigned as a new column `RAG_response`
                in the search output. If a Callable is provided, it must take one argument (`VectorStoreSearchOutput`) and
                return a valid pydantic model instance with a list attribute `answer` of the same length as the number of rows.
                If None, defaults to a pydantic model which validates the `answer` is a list of the correct length.
            model_name (str): The name of the HuggingFace model to use for RAG. Defaults to "deepseek-ai/DeepSeek-v3.2".
                Note: chosen model must be tagged with "conversational" on the model hub.
            model_provider (str): The provider of the HuggingFace model. Defaults to None.
            visible_cols (list[str]): The columns of the search output to include in the prompt for the LLM.
                Defaults to ["query_text", "doc_text"].
            **client_kwargs: Additional keyword arguments to pass to the Hugging Face Inference client.

        Raises:
            ConfigurationError: If the Hugging Face Inference client fails to initialize.
        """
        check_deps(["huggingface_hub"], extra="huggingface")
        from huggingface_hub import InferenceClient  # type: ignore

        self.model_name = model_name
        self.model_provider = model_provider
        self.context_prompt = context_prompt
        self.response_prompt = response_prompt
        self.context_message_hf = {"role": "system", "content": self.context_prompt}
        self.response_pydantic_template = llm_response_pydantic_model or self._default_LLM_response_model_builder
        self.visible_cols = visible_cols or ["query_text", "doc_text"]
        self.client_kwargs = client_kwargs

        try:
            if self.model_provider:
                self.client = InferenceClient(provider=self.model_provider, api_key=api_key)
            else:
                self.client = InferenceClient(api_key=api_key)
        except Exception as e:
            raise ConfigurationError(
                "Failed to initialize Hugging Face Inference client.",
                context={"hooks": "HuggingFaceRAG", "cause": str(e), "cause_type": type(e).__name__},
            ) from e

    @staticmethod
    def _default_LLM_response_model_builder(search_subset: VectorStoreSearchOutput) -> BaseModel:
        """Builds a default pydantic model for validating the LLM response, which checks that the response is a list of strings of the correct length."""
        row_count = len(search_subset)

        class _DefaultLLMResponseModel(BaseModel):
            answer: list | None = Field(
                default=None,
                strict=True,
                min_length=row_count,
                max_length=row_count,
                description="The LLM-generated response for a single query.",
            )

        return _DefaultLLMResponseModel

    @staticmethod
    def _response_format_builder(response_model: BaseModel):
        """Builds the response formatting instructions for the LLM, based on the fields of the provided pydantic model.

        Args:
            response_model (BaseModel): A pydantic model defining the expected schema of the LLM response.

        Returns:
            (dict): A dictionary defining the LLM response schema.
        """
        return {
            "type": "json_schema",
            "json_schema": {"name": "RAG_Response", "schema": response_model.model_json_schema(), "strict": True},
        }

    def _format_prompt_single_query(self, search_subset: VectorStoreSearchOutput) -> str:
        """Format a prompt directing the LLM to process search responses corresponding to a single origin query.
        The prompt includes instructions, search output schema, description for formatting the response, and the
        search output itself.

        Args:
            search_subset (VectorStoreSearchOutput): A subset of the search output corresponding to a single query.

        Returns:
            (str): The formatted prompt for the LLM.
        """
        return f"""
    Instructions:
    -------------
    Process the data provided in the Data section according to the task description given in the Context Description.
    Take advice from the Output Format section when formatting your response.

    Context Description:
    --------------------
    {self.context_prompt}

    Output Format:
    -------------
    The output instructions are as follows:
    {self.response_prompt}

    Data:
    -----
    {search_subset[self.visible_cols].to_string()}
        """

    def __call__(self, search_output: VectorStoreSearchOutput) -> VectorStoreSearchOutput:
        """Calls the Hugging Face Inference API to add a `RAG_response` column to the search output.

        Args:
            search_output (VectorStoreSearchOutput): The output from the `.search()` method.

        Returns:
            (VectorStoreSearchOutput): The search output with an additional column for the LLM-generated RAG response.

        Notes:
            - This method adds a new column `RAG_response` to the VectorStoreSearchOutput object. The format of the response
              is user-specified, via the `response_prompt` parameter of the hook, and is validated by the `llm_response_pydantic_model`
              parameter of the hook (defaulting to a pydantic model which checks that the response is a list of strings of the correct
              length, if omitted).
            - Each unique query in the search output is processed separately, with a prompt formatted using the
              `_format_prompt_single_query` method.
        """
        updated_search_output = search_output.copy()
        updated_search_output["RAG_response"] = ""
        distinct_queries = search_output["query_id"].unique()
        for query_id in distinct_queries:
            search_subset = search_output[search_output["query_id"] == query_id]
            prompt = {"role": "user", "content": self._format_prompt_single_query(search_subset)}
            # If custom model passed:
            if isinstance(self.response_pydantic_template, type) and issubclass(
                self.response_pydantic_template, BaseModel
            ):
                response_model = self.response_pydantic_template
            # If instance of custom model passed:
            elif isinstance(self.response_pydantic_template, BaseModel):
                response_model = self.response_pydantic_template.__class__
            # If model function/generator passed:
            else:
                response_model = self.response_pydantic_template(search_subset)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[prompt],
                max_tokens=max((1024, len(search_subset) * 50)),
                response_format=self._response_format_builder(response_model),
                temperature=0.0,
                stream=False,
                top_p=0.3,
            )
            try:
                parsed_response = response_model.model_validate_json(response.choices[0].message.content)
            except Exception:
                fake_answer = ["LLM response could not be parsed"] * len(search_subset)
                parsed_response = response_model(answer=fake_answer)  # type: ignore
            print(parsed_response.answer, type(parsed_response.answer))
            updated_search_output.loc[search_subset.index, "RAG_response"] = [str(i) for i in parsed_response.answer]
        return updated_search_output
