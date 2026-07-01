import io
import json
from collections.abc import Callable

import numpy as np
import pandas as pd

from classifai._optional import check_deps
from classifai.exceptions import ConfigurationError, HookError
from classifai.indexers.dataclasses import VectorStoreSearchOutput
from classifai.indexers.hooks.hook_factory import HookBase


class DeduplicationHook(HookBase):
    """A post-processing hook to remove duplicate knowledgebase entries.

    Deduplicates entries that share the same label within each query's
    results, aggregating their scores using the specified method.

    Attributes:
        score_aggregation_method (str): The name of the score aggregation method
            in use. Either "max" or "mean".
        score_aggregator (Callable): The callable used to aggregate scores for
            deduplicated entries.
    """

    def _mean_score(self, scores):
        """Returns the mean of the given scores."""
        return np.mean(scores)

    def _max_score(self, scores):
        """Returns the maximum of the given scores."""
        return np.max(scores)

    def __init__(self, score_aggregation_method: str = "max"):
        """Initialises the hook with the specified score aggregation method.

        Args:
            score_aggregation_method (str): Method for assigning a score to the
                deduplicated entry. Must be one of "max" or "mean". Defaults to
                "max". A future update will introduce a "softmax" option.

        Raises:
            HookError: If score_aggregation_method is not one of the
                accepted values.
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
        """Aggregates retrieved knowledgebase entries with the same label.

        For each query, deduplicates entries that share the same label,
        aggregating their scores using the specified method, and
        reassigning ranks accordingly.

        Args:
            input_data (VectorStoreSearchOutput): The search output data
                containing potentially duplicate entries to deduplicate.

        Returns:
            A new VectorStoreSearchOutput with duplicate labels per query
            collapsed into single entries, with scores aggregated and
            ranks reassigned.
        """
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
    """A post-processing hook to perform Retrieval Augmented Generation.

    Calls a generative LLM for each unique query in the search output,
    appending a RAG_response column to the result.

    Attributes:
        model_name (str): The name of the generative model to use.
        context_prompt (str): The system prompt providing context to the LLM.
        response_template (str): A template describing the expected output
            format for each row in the search results.
        llm_response_parser (Callable): A callable for parsing the raw LLM
            response string into a list of per-row values.
        client (genai.Client): The initialised GenAI client.
        config_generator (genai.types.GenerateContentConfig): The
            GenerateContentConfig class used to configure LLM calls.
        client_kwargs (dict): Keyword arguments used to initialise the GenAI
            client.
    """

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
        """Initialises the hook with the specified LLM and prompt configuration.

        Args:
            context_prompt (str): A system prompt to provide context to the
                LLM for RAG. Defaults to "".
            response_template (str): A template describing the expected output
                format for each row in the search results. Defaults to "".
            llm_response_parser (Callable[[VectorStoreSearchOutput, str], str] | None):
                A callable for parsing the LLM response string within the
                context of the search output for a single query. Defaults to
                _default_parse_LLM_response, which expects a valid JSON array
                of strings.
            project_id (str | None): The Google Cloud project ID. Must be
                provided together with location, and not alongside api_key.
                Defaults to None.
            api_key (str | None): The API key for authenticating with the
                GenAI API. Must be provided alone, not alongside project_id.
                Defaults to None.
            location (str): The Google Cloud location of the GenAI API.
                Defaults to "europe-west2".
            model_name (str): The name of the generative model. Defaults to
                "gemini-2.5-flash".
            **client_kwargs: Additional keyword arguments to pass to the
                GenAI client, e.g. vertexai=True.

        Raises:
            ConfigurationError: If both or neither of project_id and
                api_key are provided, or if the GenAI client fails to
                initialise.
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
        """Formats a prompt for the LLM to process results for a single query.

        The prompt includes instructions, the search output schema,
        output format guidance, and the search output itself.

        Args:
            search_subset (VectorStoreSearchOutput): A subset of the search
                output corresponding to a single query.
            query_id (str): The ID of the query corresponding to the search
                subset, used for prompt formatting.

        Returns:
            str: The formatted prompt string for the LLM.
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
        """Parses the LLM response as JSON, expecting a list of strings.

        Args:
            search_subset (VectorStoreSearchOutput): The search output
                corresponding to a single query, used to validate the
                response length.
            llm_response (str): The raw text response from the LLM.

        Returns:
            The parsed response as a list of strings, suitable for
            assignment to the RAG_response column.

        Raises:
            HookError: If the LLM response cannot be parsed as valid
                JSON, is not a list, or does not match the length of
                search_subset.
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

    def _call_llm(self, search_output: VectorStoreSearchOutput) -> VectorStoreSearchOutput:
        """Calls the LLM for each query in the search output and collects responses.

        Iterates over each unique query in search_output, formats a
        prompt using _format_prompt_single_query, calls the LLM, and
        parses the response using llm_response_parser.

        Args:
            search_output (VectorStoreSearchOutput): The output from the
                .search() method of the VectorStore.

        Returns:
            The search output with an additional RAG_response column
            containing the LLM-generated response for each row.
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
        """Calls the LLM to add a RAG_response column to the search output.

        Args:
            search_output (VectorStoreSearchOutput): The search output to
                augment with LLM-generated responses.

        Returns:
            The search output with an additional RAG_response column
            containing the LLM-generated response for each row.
        """
        processed_output = self._call_llm(search_output)
        return processed_output
