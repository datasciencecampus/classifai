# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v1.0.0] - 2026-03-27

### Added

- AI Agents - Hooks for using genai to perform tasks on VectorStore results.
- Hooks Framework - new framework for hooks to support premade and custom hook development.
- Server Class Features:
    - new methods for instantiating the FastAPI application and/or routing.
    - allows middleware to be used, or the routing to be attached to another FastAPI service.
- Documentation - new QuartoDocs documenting the ClassifAI package and new demo notebooks.
- Partial String matching - reverse search VectorStore method now does optional partial matching.
- Vectoriser Class - More options for instantiating HuggingFace models.

### Changed

- Datasets - updated dataset column names for v1.0.0
- Documentation - better docstrings and updated demo notebooks.
- Dataclasses - updated for more intuitive dataframe column naming.
- Server Class Refactor:
    - expanded scope of features.
    - renamed start_api method to run_server.

### Fixed
- Server hook data - hook metadata now returned in FastAPI responses.
- Reverse Search results - fixed issue where max_n_results defaulted to None causing errors.

## [v0.2.1] - 2026-02-06

### Fixed

- Dependency Bug - Removed old dependency code for data processing.

## [v0.2.0] - 2026-01-28

### Added

- Improved Vectorisers - More ways to use GcpVectoriser.
- Hooks - User defined hook functions for custom input and output VectorStore logic.
- Dataclasses - Data objects for interfacing with core VectorStore methods.
- Documentation and Demo - README updates, Jupyter Notebooks for running server, hooks and more.

## [v0.1.0] - 2025-11-04

### Added

- Vectoriser Class - Abstract base class, and GCP, HuggingFace, Ollama Vectorisers
- Vector Store Class
- REST API - FastAPI served with Uvicorn
- Documentation and Demo - README and Jupyter Notebook minimal demo with fake dataset.

