# Unit Testing Plan for Vectorisers Module
## Test Structure Overview

<b>1. Base Class Tests (test_vectoriser_base.py)</b>
* Verify VectoriserBase is abstract and cannot be instantiated
* Verify transform method is abstract
* Test that subclasses must implement transform


<b>2. HuggingFaceVectoriser Tests</b>(test_huggingface_vectoriser.py)
* Initialisation:
    * Missing dependencies raise appropriate errors
    * Valid model loads successfully
    * Invalid model name raises ExternalServiceError
    * Device selection (CPU/GPU) works correctly
    * Bad device selection raises ConfigurationError
    * trust_remote_code defaults to False
    * Custom kwargs are passed through
* Transform method:
    * Single string input converts to list and processes
    * List of strings processes correctly
    * Returns 2D numpy array
    * Output shape matches input count
    * Tokenisation failures raise VectorisationError
    * Model inference failures raise VectorisationError
    * Pooling failures raise VectorisationError


<b>3. GcpVectoriser Tests (test_gcp_vectoriser.py)</b>
* Initialization:
    * Missing dependencies raise appropriate errors
    * project_id + location authentication works
    * api_key authentication works
    * Missing both auth methods raises ConfigurationError
    * Providing both auth methods raises ConfigurationError
    * Client initialisation failures raise ConfigurationError
* Transform method:
    * Single string input converts to list
    * List processes correctly
    * Returns 2D numpy array
    * Output shape matches input count
    * API request failures raise ExternalServiceError
    * Unexpected response format raises VectorisationError


<b>4. OllamaVectoriser Tests (test_ollama_vectoriser.py)</b>
* Initialization:
    * Missing dependencies raise appropriate errors
    * Model name is stored correctly
* Transform method:
    * Single string input converts to list
    * List processes correctly
    * Returns 2D numpy array
    * Service failures raise ExternalServiceError
    * Response parsing failures raise VectorisationError



## Key Testing Considerations


| **Aspect**            | **Strategy**                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| External Dependencies  | Use `pytest-mock` or `unittest.mock` to patch external libraries (torch, transformers, ollama, google.genai) |
| GPU/Device Testing     | Mock `torch.cuda` to test both CPU and GPU branches if we are concerned with GPU compatibility |
| API Responses          | Mock service responses with realistic embedding data                       |
| Error Cases            | Test each exception path in try-except blocks                              |
| Input Validation       | Test both string and list inputs                                           |
| Output Validation      | Verify numpy array shape, dtype, and content                               |
