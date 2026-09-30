# Unit Testing Plan for Indexers Module
## Test Structure Overview
<b>1. Dataclass Tests (test_indexers_dataclasses.py) </b>
* VectorStoreSearchInput:
    * Valid dict/DataFrame converts and validates correctly
    * Schema validation enforces column types
    * Missing required columns raises validation error
    * Type coercion works (strings, etc.)
    * Property accessors work (id, query) / return correct series
    * Empty inputs handled correctly
* VectorStoreSearchOutput:
    * Valid construction from dict/DataFrame
    * Schema validation enforces column types
    * Missing required columns raises validation error
    * Rank column must be non-negative
    * Score column accepts floats
    * Property accessors work / return correct series
    * Column ordering is preserved (queries broadcast down consecutive rows)
    * Empty inputs handled correctly
* VectorStoreEmbedInput/Output:
    * Valid dict/DataFrame construction and validation
    * Type coercion for id/text
    * Embedding column accepts numpy arrays
    * Empty Inputs/Output handled correctly
* VectorStoreReverseSearchInput/Output:
    * Valid construction from dicts/DataFrames
    * Empty Input/Output DataFrame handles correctly
    * Schema validation works
    * Property accessors function properly
    * Missing required columns raises validation error

<b>2. VectorStore Initialization Tests (test_vectorstore_init.py) </b>
* Input validation (DataValidationError):
    * file_name must be non-empty string
    * data_type validation (only "csv" supported)
    * vectoriser must be VectoriserBase instance
    * batch_size must be positive integer
    * meta_data must be dict or None
    * hooks must be dict or None
    * output_dir must be string or None
* File system handling (ConfigurationError):
    * Input file must exist
    * Output directory creation works
    * overwrite flag prevents accidental overwrites
    * gs:// paths require gcsfs (helpful error message)
    * Invalid fsspec paths raise ConfigurationError
* Index building (IndexBuildError):
    * CSV file reads correctly
    * UUID generation works
    * Batch processing of embeddings
    * Vectoriser failures wrapped appropriately
    * Embeddings count matches batch size
    * Metadata serialization to JSON
    * Parquet file writing
* skip_save flag:
    * When True, no files written to disk
    * When False, metadata.json and vectors.parquet created
    * warning logged when output_dir set but skip_save=True

<b>3. VectorStore Search Tests (test_vectorstore_search.py)</b>
* Input validation (DataValidationError):
    * query must be VectorStoreSearchInput
    * n_results must be int >= 1
    * batch_size must be int >= 1 or None
    * Empty query raises error
    * Vector store not initialized raises ConfigurationError
* Search operation:
    * Single query processes correctly
    * Multiple queries in batch
    * Similarity scores computed (dot-product)
    * Top n_results returned per query
    * Results ranked by score (descending)
    * Output shape matches expected (n_queries * n_results rows)
    * Metadata columns included in output
    * Query batching with custom batch_size works
* Error handling (VectorisationError/ClassifaiError):
    * Query embedding failure
    * Vectoriser.transform() exceptions wrapped
    * Error context includes vectoriser class, batch info
* Hooks integration:
    * search_preprocess hook called before search
    * search_postprocess hook called after search
    * Hook failures raise HookError
    * Multiple hooks in list processed in order
    * Single hook converted to list automatically

<b>4. VectorStore Reverse Search Tests (test_vectorstore_reverse_search.py)</b>
* Input validation (DataValidationError):
    * query must be VectorStoreReverseSearchInput
    * max_n_results must be int >= 1 or -1
    * Empty query raises error
* Reverse search operation:
    * Exact label matching works (default)
    * Partial matching (prefix) when enabled
    * max_n_results limits results per query
    * max_n_results=-1 returns all matches
    * Results include metadata columns
    * Empty result sets handled (returns empty DataFrame with correct schema)
    * Sorting by id and label works
* Error handling:
    * Vectoriser-independent (no embeddings needed)
    * DataFrame join failures wrapped
    * Error context includes max_n_results, query count
* Hooks integration:
    * reverse_search_preprocess hook calls before reverse search
    * reverse_search_postprocess hook calls after reverse search
    * Same checks and error handling as search

<b>5. VectorStore Embed Tests (test_vectorstore_embed.py)</b>
* Input validation (DataValidationError):
    * query must be VectorStoreEmbedInput
    * Invalid input type raises error
* Embedding operation:
    * Single text embeds correctly
    * Multiple texts process correctly
    * Output includes id, text, and embedding
    * Embeddings are numpy arrays
    * Output shape matches input count
    * Vectoriser.transform() called with correct texts
* Error handling (VectorisationError/ClassifaiError):
    * Vectoriser failures wrapped with context
    * Error includes vectoriser class, text count
* Hooks integration:
    * embed_preprocess hook called before embedding
    * embed_postprocess hook called after embedding
    * Same checks and error handling as search

<b>6. VectorStore Metadata Tests (test_vectorstore_metadata.py)</b>
* Metadata serialization (_save_metadata):
    * JSON file created at correct path
    * Contains all required fields (vectoriser_class, vector_shape, num_vectors, batch_size, created_at, meta_data)
    * Type information preserved (str types → string names)
    * Valid JSON format
    * fsspec paths work (gs://, etc.)
* Metadata loading (from_filespace):
    * Metadata file read and parsed correctly
    * Required keys validated
    * Type deserialization works
    * Backwards compatibility with v1.0.0 (missing batch_size)
    * Default batch_size used when missing
    * Warning logged for missing batch_size

<b>7. VectorStore from_filespace Tests (test_vectorstore_from_filespace.py)</b>
* Input validation (DataValidationError):
    * folder_path must be non-empty string
    * folder_path must be existing directory
    * batch_size override must be int >= 1 or None
    * hooks must be dict or None
* File loading (IndexBuildError):
    * metadata.json exists and valid
    * vectors.parquet exists and valid
    * Required columns present in parquet
    * Parquet not empty
    * Metadata can be deserialized
* Configuration validation (ConfigurationError):
    * Vectoriser class name matches metadata
    * vectoriser must have callable .transform() method / inherit from base class
    * fsspec paths (gs://) work with gcsfs
    * Helpful error message when gcsfs missing
* Instance construction:
    * Instance created without calling init
    * All attributes set correctly
    * batch_size override works
    * metadata.meta_data deserialized and set
    * Vectoriser instance attached
    * hooks parameter applied
    * quiet_mode applied
    * Instance is functional (can search/embed/reverse_search)


## Key Testing Considerations
| **Aspect**              | **Strategy**                                                                                                                                               |
|--------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| Vectoriser Mocking       | Mock `VectoriserBase` to return predictable embeddings; separate vectoriser testing from vectorstore testing                                              |
| File System              | Mock `fsspec` for local/remote paths; test real local paths in integration tests; `gs://` tests optional/skipped without `gcsfs`                          |
| Dataclass Validation     | Test both valid and invalid inputs; verify `pandera` schema enforcement; test type coercion                                                              |
| Large Datasets           | Use small synthetic CSVs (< 100 rows); mock large searches with synthetic embeddings to avoid slow tests                                                 |
| Similarity Computation   | Verify dot-product calculations; test edge cases (zero embeddings, identical embeddings, single query)                                                   |
| Hook System              | Mock hooks that modify input/output; test hook chains; verify error propagation; test that single hooks auto-convert to lists                            |
| Error Context            | Verify all exceptions include relevant context (vectoriser class, batch info, file paths) without exposing secrets                                       |
| Save/Load Cycle          | Test round-trip (create → save → load); verify metadata preservation; test backwards compatibility with old metadata format                               |
| Empty/Edge Cases         | Empty query results, single document, single query, all identical embeddings, `max_n_results > available docs`                                           |
| Quiet Mode               | Verify progress bars suppressed; verify logging levels adjusted; test both `True` and `False` paths                                                     |
