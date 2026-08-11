# Unit Testing Plan for Evaluation Module
## Test Structure Overview
<b>1. parse_metrics Function Tests (test_parse_metrics.py)</b>
* Valid metric parsing:
    * Single valid metric name returns dict with one entry
    * Multiple valid metric names return dict with all entries
    * Case insensitivity (accepts "accuracy", "ACCURACY", "Accuracy")
    * Metric instances are correct type (e.g., ClassificationAccuracy for "accuracy")
    * Dict keys match input metric names (lowercase)
    * Dict values are Metric instances with evaluate() method
* Invalid metric handling (ValueError):
    * Single invalid metric name raises ValueError
    * Invalid metric in list of valid ones raises error
    * Error message includes the invalid metric name
    * Error message includes list of valid metrics
    * Empty string metric name raises error
    * Whitespace-only metric name raises error
    * Typos caught (e.g., "accuraccy" instead of "accuracy")
* Edge cases:
    * Empty list returns empty dict
    * Duplicate metric names handled (e.g., ["accuracy", "accuracy"] returns dict with 1 entry)
    * Mixed case like "MacRo_F1" handled correctly

<b>2. Evaluation Initialization Tests (test_evaluation_init.py)</b>
* Input validation (ground_truths DataFrame):
    * DataFrame with correct schema (text, label columns) validates
    * Missing 'text' column raises pandera SchemaError
    * Missing 'label' column raises pandera SchemaError
    * Wrong dtype for 'text' (not string) raises pandera error
    * Wrong dtype for 'label' (not string) raises pandera error
    * Type coercion works (int/float converted to string)
    * Empty DataFrame validates (0 rows but correct schema)
    * Extra columns allowed but ignored
* Metrics validation (InvalidMetricError):
    * Valid metric list ["accuracy", "macro_f1"] parses successfully
    * Invalid metric name raises InvalidMetricError
    * Empty metrics list accepted (no metrics to compute)
    * InvalidMetricError has code "invalid_metric_error"
    * InvalidMetricError context includes metrics list and cause
* Batch size validation (DataValidationError):
    * Default batch_size is correct
    * Custom batch_size (e.g., 16, 32) stored correctly
    * Negative batch_size raises error (if checked in init)
    * Zero batch_size raises error (if checked in init)
* save_output flag:
    * Default is False
    * Can be set to True
    * Boolean type enforced
* Attributes set correctly:
    * self.ground_truths is a copy (not reference)
    * self.ground_truths has new 'qid' column added (index as string)
    * self.batch_size set correctly
    * self.save_output set correctly
    * self.metric_results initialized as empty dict
    * self.parsed_metrics is dict of Metric instances
* qid column generation:
    * All rows get unique qid values
    * qid values are strings
    * qid values correspond to original index
    * Original ground_truths data unchanged

<b>3. Evaluation.evaluate() Method Tests (test_evaluation_evaluate.py)</b>
* Input validation:
    * vectorstores must be list (not tuple, dict)
    * vectorstore_names must be list (not tuple, dict)
    * Length of vectorstores must equal length of vectorstore_names
    * Each item in vectorstores is VectorStore instance or callable
    * Invalid VectorStore instance at index i caught with context
    * All vectorstore_names must be strings
    * Invalid name type at index i caught
    * All vectorstore_names must be unique (no duplicates)
    * output_file must be string or None
    * output_file must end with ".csv" if provided
    * overwrite must be boolean
* File system handling (save_output=True):
    * Default output_file is "evaluation_results.csv" if save_output=True and output_file=None
    * Existing file raises error if overwrite=False
    * Existing file overwritten if overwrite=True
    * Parent directories created if don't exist
    * Directory creation errors handled
    * Results saved to correct file path
    * CSV format is correct (columns: vectorstore_name, metric names)
* File system handling (save_output=False):
    * No file written even if output_file provided
    * No error raised for existing files
* VectorStore processing:
    * Each vectorstore processed sequentially
    * Callable vectorstores instantiated before use
    * Callable instantiation errors wrapped in EvaluationError
    * Instance vectorstores used directly
    * Invalid callable (doesn't return VectorStore) caught
    * VectorStore deleted from memory after use if callable
* Search execution (_run_search):
    * VectorStoreSearchInput created with qid and text columns
    * vectorstore.search() called with correct params (n_results=1)
    * batch_size from Evaluation passed to search
    * Search failure wrapped in EvaluationError with context
    * Search error context includes vectorstore_name
* Results validation:
    * _run_search returns DataFrame with SearchOutputSchema
    * Required columns present (query_id, query_text, doc_label, doc_text, rank, score, ground_truth_label)
    * ground_truth_label column merged correctly from ground_truths
    * Schema validation enforces column types and constraints
    * rank >= 0 enforced
    * Pandera validation failures raised as SchemaError
* Metric computation:
    * Each parsed metric evaluated on results
    * metric.evaluate() called with results DataFrame
    * Metric results stored in self.metric_results
    * Metric computation errors wrapped in EvaluationError
    * Error context includes vectorstore_name and cause
    * Metric results persist across vectorstores (accumulate)
* Results aggregation:
    * DataFrame created for each vectorstore with metric results
    * Row indexed by vectorstore name
    * All metrics included as columns
    * Overall DataFrame concatenates results from all stores
    * Row order matches vectorstore order
    * No duplicate rows
* Error handling and cleanup:
    * Any step failure raises appropriate exception type
    * Errors include context (vectorstore_name, cause)
    * Processing continues until error (no partial results)
    * Callable vectorstores cleaned up even on error (finally block)
* Return value:
    * Returns DataFrame with vectorstore names as index
    * One row per vectorstore
    * Columns are metric names
    * Values are floats
* Edge cases:
    * Single vectorstore works
    * Many vectorstores work (10+)
    * Empty ground_truths (0 queries) handled
    * Mixed callables and instances in same list

<b>4. Evaluation._run_search() Method Tests (test_evaluation_run_search.py)</b>
* Search input construction:
    * VectorStoreSearchInput created with correct data
    * 'id' column from self.ground_truths['qid']
    * 'query' column from self.ground_truths['text']
    * Input order matches ground_truths order
* Search execution:
    * vectorstore.search() called with SearchInput
    * n_results=1 passed (top-1 search)
    * batch_size from self.batch_size passed
    * Search result is DataFrame
* Merge operation:
    * Results merged with ground_truths on query_id → qid
    * Left join preserves all results
    * Merge columns: qid, label from ground_truths
    * 'label' column renamed to 'ground_truth_label'
    * Column order correct in output
* Output validation:
    * Returns DataFrame with SearchOutputSchema
    * All required columns present
    * Pandera validation passed
    * No extra/unexpected columns (or handled gracefully)
    * Row count matches input queries
* Error handling:
    * VectorStore errors not caught (because they propagate to evaluate()
    * Pandera validation errors propagate (SearchOutputSchema)
    * Merge errors propagate (shouldn't happen with valid input)
* Edge cases:
    * Single query works
    * Many queries work
    * Query with no matching results handled (None/NaN for ground_truth_label)

<b>5. Metric Base Class Tests (test_metrics_base.py)</b>
* Metric ABC enforcement:
    * Cannot instantiate Metric directly (abstract)
    * Subclasses must implement evaluate()
    * Subclass missing evaluate() raises TypeError
* MetricResult dataclass:
    * Can be instantiated with name and value
    * Has repr that formats as "name: value"
    * Value formatted to 4 decimal places

<b>6. ClassificationAccuracy Metric Tests (test_metrics_accuracy.py)</b>
* Correct predictions:
    * All correct predictions → accuracy = 1.0
    * No correct predictions → accuracy = 0.0
    * 50% correct → accuracy = 0.5
    * Accuracy = correct_count / total_count
* Edge cases:
    * Empty DataFrame (0 rows) → 0.0 (or error?)
    * Single prediction correct → 1.0
    * Single prediction wrong → 0.0
    * NaN/None values handled
    * Case sensitivity (doc_label vs ground_truth_label)
* Output:
    * Returns MetricResult
    * name = "accuracy"
    * value is float in [0.0, 1.0]

<b>7. ClassificationMacroRecall Metric Tests (test_metrics_macro_recall.py)</b>
* Single label:
    * All true positives → recall = 1.0
    * No true positives → recall = 0.0
    * Recall = TP / (TP + FN) per label
* Multiple labels:
    * Recalls computed per label
    * Macro recall is average of per-label recalls
    * Recall for unseen label is 0.0
    * Recall with zero denominator (TP=0, FN=0) is 0.0
* Edge cases:
    * Empty DataFrame → 0.0
    * Single label → that label's recall
    * Labels only in predictions (not ground truth) → FN=0, Recall=1.0
    * Labels only in ground truth (not predictions) → TP=0, FN>0, Recall=0.0
* Output:
    * Returns MetricResult with name "macro_recall"
    * Value is float in [0.0, 1.0]

<b>8. ClassificationMacroPrecision Metric Tests (test_metrics_macro_precision.py)</b>
* Single label:
    * All true positives → precision = 1.0
    * No true positives → precision = 0.0
    * Precision = TP / (TP + FP) per label
* Multiple labels:
    * Precisions computed per label
    * Macro precision is average of per-label precisions
    * Precision for unseen label is 0.0
    * Precision with zero denominator (TP=0, FP=0) is 0.0
* Edge cases:
    * Empty DataFrame → 0.0
    * Single label → that label's precision
    * False positives only → precision = 0.0
* Output:
    * Returns MetricResult with name "macro_precision"
    * Value is float in [0.0, 1.0]

<b>9. ClassificationMacroF1 Metric Tests (test_metrics_macro_f1.py)</b>
* F1 calculation:
    * F1 = 2 * (precision * recall) / (precision + recall)
    * F1 with zero denominator = 0.0
    * Perfect precision and recall → F1 = 1.0
    * Zero precision and recall → F1 = 0.0
* Multiple labels:
    * F1 computed per label
    * Macro F1 is average of per-label F1s
    * F1 for unseen label is 0.0
* Edge cases:
    * Empty DataFrame → 0.0
    * Single label → that label's F1
    * Precision = 0, Recall > 0 → F1 = 0.0
* Output:
    * Returns MetricResult with name "macro_f1"
    * Value is float in [0.0, 1.0]

<b>10. Schema Validation Tests (test_evaluation_schemas.py)</b>
* GroundTruthSchema:
    * Accepts 'text' and 'label' columns (string type)
    * Coerces types (int/float → string)
    * Rejects missing columns
    * Rejects wrong dtypes (without coercion possible)
    * Extra columns allowed
* SearchOutputSchema:
    * All required columns present
    * Correct dtypes
    * rank >= 0 constraint enforced
    * score is float
    * Coercion applied
    * Rejects missing columns
    * Rejects invalid types

<b>11. Other Edge Cases and Error Handling (test_evaluation_edge_cases.py)</b>
* DataFrame edge cases:
    * 0 queries (empty ground_truths)
    * 1 query
    * Many queries (1000+)
    * Special characters in labels
    * Very long text strings
    * Unicode in labels/text
    * Whitespace in labels
* Metric edge cases:
    * All predictions correct
    * All predictions wrong
    * Perfect imbalance (1 label dominates)
    * Many labels (10+)
    * Predictions never match ground truth
    * Labels only exist in predictions
    * Labels only exist in ground truth
* Exception scenarios:
    * VectorStore raises exception during search
    * Callable raises exception during instantiation
    * Metric raises exception during evaluate
    * File save fails (permission error)
    * Merge operation produces unexpected shape
    * Pandera validation fails on search results



## Key Testing Considerations

| **Aspect**              | **Strategy**                                                                                                                                                                                                                     |
|--------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| VectorStore Mocking      | Mock VectorStore.search() to return realistic SearchOutputSchema DataFrames; control results to test different accuracy/metric scenarios                                                                                      |
| DataFrame Creation       | Create realistic ground_truths and search result DataFrames with all required columns; test both happy path and edge cases                                                                                                   |
| Pandera Validation       | Test schema acceptance/rejection; verify type coercion works; test constraint enforcement (rank >= 0)                                                                                                                         |
| Callable Handling        | Mock callables that return VectorStore instances; test callable errors separately from VectorStore errors; verify cleanup in finally block                                                                                   |
| Metric Computation       | Mock metrics to control results; test metric.evaluate() called with correct data; verify MetricResult objects correct                                                                                                        |
| File I/O                 | Mock filesystem operations to avoid creating real files in tests; test path logic (directory creation, CSV format) separately from core logic                                                                                |
| Integration Flow         | Test full workflow with real (small) DataFrames and mocked VectorStore; verify data flows through each step correctly                                                                                                        |
| Error Propagation        | Verify exceptions wrapped with correct context; test error messages include vectorstore_name; verify cleanup happens even on error                                                                                           |
| Metric Formulas          | Hand-calculate expected metric values for simple test cases; verify computed values match; test edge cases (zero denominators, NaN, etc.)                                                                                    |
| Label Handling           | Test with various label distributions (balanced, imbalanced, single label, many labels); test missing/NaN labels                                                                                                            |
| Case Sensitivity         | Test that column names are matched correctly (doc_label vs ground_truth_label); test label values are case-sensitive                                                                                                         |
| Type Coercion            | Verify Pandera coercion works (int→str, float→str); test when coercion would fail (e.g., complex objects)                                                                                                                     |
| Performance              | Don't test with huge datasets (1M rows); use small (10-100 row) test DataFrames; integration tests run quickly                                                                                                               |

