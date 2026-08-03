# Unit Testing Plan for Servers Module
## Test Structure Overview
<b>1. Router Creation Tests (test_get_router.py)</b>
* Input validation (DataValidationError):
    * vector_stores must be a list (not tuple, dict, etc.)
    * endpoint_names must be a list (not tuple, dict, etc.)
    * Length of vector_stores must match length of endpoint_names
    * All endpoint_names must be non-empty strings
    * No whitespace-only strings allowed
    * endpoint_names must be unique (no duplicates)
    * Empty lists raise appropriate error
* VectorStore validation (ConfigurationError):
    * Each item in vector_stores must be VectorStore instance
    * Invalid type at specific index raises error with context
    * Mixed valid/invalid stores caught at first invalid
    * Error context includes the invalid index and type
* Router creation:
    * Router is successfully created and returned
    * Router is FastAPI APIRouter instance
    * Endpoints are registered for each vector store
    * Correct number of sub-routers created
    * Docs endpoint "/" redirects to "/docs"
    * Router has correct tags for each endpoint
* Edge cases:
    * Single vector store works
    * Many vector stores work (10+)
    * Special characters in endpoint names handled
    * Case sensitivity in endpoint names preserved
    
<b>2. Server Creation Tests (test_get_server.py)</b>
* Input validation (delegates to get_router):
    * Same validation as get_router tested indirectly
    * Invalid inputs raise same errors
* FastAPI app creation:
    * FastAPI instance returned
    * App title set correctly ("ClassifAI API Server")
    * App description set correctly
    * App version matches __version__
    * OpenAPI tags created for each endpoint name
    * Each tag has correct name and description
    * Router included in app
    * Docs and redoc endpoints available
* Integration:
    * Router endpoints accessible through app
    * All three endpoint types (search, embed, reverse_search) present
    * Correct number of paths registered
    * Tags properly organized in OpenAPI spec

<b>3. Server Runtime Tests (test_run_server.py)</b>
* Input validation (DataValidationError):
    * port must be integer
    * port must be >= 1
    * port must be <= 65535
    * Negative port raises error
    * Port 0 raises error
    * Port 65536 raises error
    * host_ip must be string
    * Empty host_ip validation
* Log level validation (DataValidationError):
    * Valid log levels: "debug", "info", "warning", "error", "critical"
    * Invalid log level raises error
    * Case sensitivity (lowercase required)
    * Error message includes valid options
    * Empty string raises error
* Server startup (mocked uvicorn):
    * uvicorn.run() called with correct parameters
    * Port passed correctly
    * Host IP passed correctly
    * Log level passed correctly
    * App created before uvicorn.run()
    * Correct number of calls to uvicorn.run()
* demo_mode flag:
    * When False, app title remains "ClassifAI API Server"
    * When True, app title changes to "ClassifAI API Demo Server"
    * When True, app description changes to demo description
    * _set_demo_defaults() called only when True
    * Other app settings unaffected by demo_mode
* Error handling:
    * Invalid port caught before uvicorn.run()
    * Invalid log_level caught before uvicorn.run()
    * Validation errors propagate correctly

<b>4. Endpoint Creation Tests (test_endpoint_creation.py)</b>
* Search endpoint creation (_create_search_endpoint):
    * Endpoint registered at /{name}/search
    * HTTP method is POST
    * Endpoint summary includes endpoint name
    * Endpoint description includes endpoint name
    * n_results query parameter has correct constraints (ge=1)
    * n_results default value is 10
    * Endpoint callable and returns SearchResponseBody
* Embed endpoint creation (_create_embedding_endpoint):
    * Endpoint registered at /{name}/embed
    * HTTP method is POST
    * Endpoint summary includes endpoint name
    * Endpoint description includes endpoint name
    * No query parameters
    * Endpoint callable and returns EmbedResponseBody
* Reverse search endpoint creation (_create_reverse_search_endpoint):
    * Endpoint registered at /{name}/reverse_search
    * HTTP method is POST
    * Endpoint summary includes endpoint name
    * Endpoint description includes endpoint name
    * max_n_results query parameter with correct constraints
    * max_n_results can be -1 (return all) or >= 1
    * max_n_results default value is 100
    * partial_match query parameter is boolean
    * partial_match default is False
    * Manual validation for max_n_results < 1 (when != -1) raises HTTPException(422)
    * Endpoint callable and returns ReverseSearchResponseBody

<b>5. Endpoint Functional Tests (test_endpoint_functionality.py)</b>
* Search endpoint functionality:
    * Extracts ids and queries from request
    * Creates VectorStoreSearchInput with correct data
    * Calls vectorstore.search() with correct params
    * Calls convert_search_dataframe_to_pydantic_response()
    * Returns formatted result as JSON
    * Vectorstore search failure propagates as 500
    * Invalid input format raises 422
    * Empty queries handled
* Embed endpoint functionality:
    * Extracts ids and texts from request
    * Creates VectorStoreEmbedInput with correct data
    * Calls vectorstore.embed() with correct params
    * Calls convert_embedding_dataframe_to_pydantic_response()
    * Returns formatted result as JSON
    * Vectorstore embed failure propagates
    * Invalid input format raises 422
    * Empty texts handled
* Reverse search endpoint functionality:
    * Extracts ids and doc_labels from request
    * Creates VectorStoreReverseSearchInput with correct data
    * Calls vectorstore.reverse_search() with correct params
    * Calls convert_reverse_search_dataframe_to_pydantic_response()
    * Returns formatted result as JSON
    * max_n_results validation (< 1 when != -1) raises HTTPException(422)
    * Vectorstore reverse_search failure propagates
    * Invalid input format raises 422

<b>6. Response Conversion Tests (test_response_conversions.py)</b>
* Search response conversion (convert_search_dataframe_to_pydantic_response):
    * Valid DataFrame converts to SearchResponseBody
    * Grouped by query_id correctly
    * Each group becomes SearchResponseSet
    * Required columns present in output (query_id, query_text, entries)
    * Metadata columns extracted and included dynamically
    * Hook columns identified and included dynamically
    * Rank column included (0-indexed or 1-indexed depending on implementation)
    * Score column included as float
    * Empty groups handled
    * Multiple queries grouped separately
    * Metadata dict respected (only columns in meta_data included)
* Reverse search response conversion (convert_reverse_search_dataframe_to_pydantic_response):
    * Valid DataFrame converts to ReverseSearchResponseBody
    * Includes original_input to ensure all inputs in response
    * Grouped by input id
    * Empty result sets for inputs with no matches (still included in response)
    * Each group becomes ReverseSearchResponseSet
    * Required columns present (input_id, searched_doc_label, entries)
    * Metadata columns extracted and included dynamically
    * Hook columns identified and included dynamically
    * searched_doc_label taken from first row of group
    * doc_label and doc_text included in entries
    * Multiple inputs handled separately
* Embed response conversion (convert_embedding_dataframe_to_pydantic_response):
    * Valid DataFrame converts to EmbedResponseBody
    * Each row becomes EmbedResponseEntry
    * id column included
    * text column included
    * embedding column converted to list (numpy array → list)
    * Hook columns identified and included dynamically
    * No meta_data parameter required
    * Multiple embeddings handled
    * Empty DataFrame returns empty data list
    * Embedding dtype preserved (floats)

<b>7. Make Endpoints Tests (test_make_endpoints.py)</b>
* Router/app routing (make_endpoints):
    * Accepts APIRouter or FastAPI app
    * Creates sub_router for each vector store
    * Sub_router has correct prefix (/{name})
    * Sub_router has correct tags ([name])
    * All three endpoint types created for each store
    * Sub_routers included in main router/app
    * Correct number of total endpoints
    * Endpoints accessible at correct paths

## Key Testing Considerations
| Aspect                | Strategy                                                                                                                                                                                                 |
|-----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| VectorStore Mocking   | Mock VectorStore instances with search/embed/reverse_search methods returning realistic DataFrames; avoid actual vectorstore initialization                                                             |
| FastAPI Testing       | Use TestClient from starlette to make HTTP requests; test actual endpoint behavior, not just function calls                                                                                            |
| Pydantic Models       | Verify request models parse correctly; test both valid and invalid JSON payloads; verify response models serialize correctly                                                                           |
| Input Validation      | Test all validation branches in get_router/get_server/run_server; verify context information in errors                                                                                                |
| Query Parameters      | Test ge/le constraints on query parameters; test default values; test invalid type coercion                                                                                                           |
| DataFrame Conversions | Use real pandas DataFrames with all required columns; test edge cases (empty, single row, many rows); verify column ordering preserved                                                                |
| Metadata Handling     | Mock meta_data dict; verify only specified columns included; test missing metadata columns gracefully ignored                                                                                         |
| Hook Columns          | Test DataFrames with extra columns; verify they're included in responses; test without hook columns                                                                                                   |
| HTTPException Handling| Mock vectorstore to raise exceptions; verify they propagate correctly; test manual validation (e.g., max_n_results check)                                                                             |
| URL Construction      | Verify endpoint paths are correct (/{name}/search, etc.); test special characters in endpoint names; test path collision prevention                                                                   |
| Logging               | Mock logger; verify appropriate logs at startup (Starting ClassifAI Router, Generating ClassifAI API, Registering endpoints)                                                                         |
| Demo Mode             | Test both True/False branches; verify only title/description changed, not functionality                                                                                                               |

