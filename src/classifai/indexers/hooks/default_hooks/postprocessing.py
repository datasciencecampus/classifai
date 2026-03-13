import numpy as np
import pandas as pd

from classifai.exceptions import HookError
from classifai.indexers.dataclasses import VectorStoreSearchOutput
from classifai.indexers.hooks.hook_factory import HookBase


class DeduplicationHook(HookBase):
    """A pre-processing hook to remove duplicate knowledgebase entries, i.e. entries with the same label."""

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
        # Softmax not supported until normalisation is implemented.
        # elif self.score_aggregation_method == "softmax":
        #     self.score_aggregator = ...
        elif self.score_aggregation_method == "mean":
            self.score_aggregator = self._mean_score

        super().__init__(hook_type="post_processing")

    def __call__(self, input_data: VectorStoreSearchOutput) -> VectorStoreSearchOutput:
        """Aggregates retrieved knowledgebase entries corresponding to the same label."""
        df_gpby = (
            input_data.groupby(["query_id", "query_text", "doc_id"])
            .aggregate(
                score=("score", self.score_aggregator),
                idxmax=("score", "idxmax"),
                rank=("rank", "min"),
            )
            .reset_index()
        )

        for query in df_gpby["query_id"].unique():
            batch = df_gpby[df_gpby["query_id"] == query]
            new_rank = pd.factorize(-batch["score"], sort=True)[0] + 1
            df_gpby.loc[batch.index, "rank"] = new_rank

        for col in set(input_data.columns).difference(set(df_gpby.columns)):
            df_gpby[col] = df_gpby["idxmax"].map(input_data[col])

        processed_output = input_data.__class__.validate(df_gpby[input_data.columns])
        return processed_output
