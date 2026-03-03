import logging

from classifai._optional import check_deps
from classifai.exceptions import ConfigurationError
from classifai.indexers.dataclasses import VectorStoreSearchOutput

try:
    from google import genai
except ImportError:
    check_deps(["google-genai"], extra="gcp")

from .hook_factory import PostProcessingHookBase

logging.getLogger("google.auth").setLevel(logging.WARNING)
logging.getLogger("google.cloud").setLevel(logging.WARNING)
logging.getLogger("google.api_core").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class RAGHook(PostProcessingHookBase):
    def __init__(  # noqa: PLR0913
        self,
        context_prompt: str = "",
        response_template: str = "",
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
            project_id (str): [optional] The Google Cloud project ID. Defaults to None.
            api_key (str): [optional] The API key for authenticating with the GenAI API. Defaults to None.
            location (str): [optional] The location of the GenAI API. Defaults to None.
            model_name (str): [optional] The name of the generative model. Defaults to "gemini-2.5-flash".
            **client_kwargs: Additional keyword arguments to pass to the GenAI client,
                             e.g. vertexai=True.

        Raises:
            ConfigurationError: If the GenAI client fails to initialize.
        """
        self.model_name = model_name
        self.context_prompt = context_prompt
        self.response_template = response_template

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

        self._setup_rag_hook()

    def _setup_rag_hook(self) -> None:
        """Set up for the RAG hook.

        Args:
            model_name (str): The name of the LLM model to use.

        Raises:
            ConfigurationError: If the GCP genai client fails to initialise.
        """
        try:
            self.client = genai.Client(**self.client_kwargs)  # .aio
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
        return f"""
        Instructions:
        -------------
        Process the data provided in the Data section (format described in the Input Format section) according to the
        task description given in your system prompt. Use the Output Format section to format your response.

        Input Format:
        -------------
        The Data will be a Pandas DataFrame, converted to JSON. The schema of the DataFrame is described as follows:
        {search_subset.head(n=0).info(verbose=True, show_counts=False, memory_usage=False)}

        Output Format:
        --------------
        The output, for each row in the Data section, should be formatted as follows:
        {self.response_template}

        Data:
        -----
        {search_subset.to_json()}
        """

    def _call_llm(self, search_output: VectorStoreSearchOutput) -> str:
        """Calls the LLM to generate responses for each query in the search output, using the formatted prompts.

        Args:
            search_output (VectorStoreSearchOutput): The output from the `.search()` method.

        Returns:
            (VectorStoreSearchOutput): The search output with an additional column for the LLM-generated RAG response.

        Notes:
            - This method adds a new column `RAG_response` to the VectorStoreSearchOutput object. The format of the response
              is user-specified, via the `response_template` parameter of the hook.
            - Each unique query in the search output is processed separately, with a prompt formatted using the
              `_format_prompt_single_query` method.
        """
        updated_search_output = search_output.copy()
        updated_search_output["RAG_response"] = ""
        distinct_queries = search_output["query_id"].unique()
        for query_id in distinct_queries:
            search_subset = search_output[search_output["query_id"] == query_id]
            prompt = self._format_prompt_single_query(search_subset, query_id)
            updated_search_output.loc[search_subset.index, "RAG_response"] = (
                self.client.models.generate_content(  # await ...
                    model=self.model_name,
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(system_instruction=self.context_prompt),
                )
            )
        return updated_search_output

    def __call__(self, search_output: VectorStoreSearchOutput) -> VectorStoreSearchOutput:
        """Calls the LLM to add the `RAG_response` column."""
        processed_output = search_output.validate(self._call_llm(search_output))
        return processed_output
