# Contributing to ClassifAI

👍🎉 Thank you for considering contributing to the Classifai project! 🎉👍

Please take a moment to review these guidelines before you start helping out.


## What should I know before getting started with contributions?
ClassifAI is a Python package designed to simplify a pipeline of text embedding, vector search, and deployment tasks. We encourage and appreciate contributions, whether it's asking questions about usage, reporting bugs, submitting fixes, or proposing new features. Your involvement helps improve the library and makes it more robust and user-friendly for everyone.😊


## Quick links
- You can find the main repository's [Readme file here](./README.md) for more details about the project (just in case you're very lost).
- Have a look at existing bugs and issues that are monitored on the [GitHub Issues page](https://github.com/datasciencecampus/classifai/issues).
- Check out the <b>Development Setup</b> section in the Readme file to quickly set up your environment for development.


## How to Contribute
### Got a question about ClassifAI?
If you have a question about how to use ClassifAI, encounter unexpected behavior, or need clarification on something, feel free to [open an issue](https://github.com/datasciencecampus/classifai/issues). When asking a question:

- Check the [README file](./README.md), DEMO ipynb [notebooks](./DEMO/), and existing [issues page](https://github.com/datasciencecampus/classifai/issues) to see if your question has already been answered.
- Use a clear and descriptive title for your issue.
- Provide as much context as possible, including:
  - What you are trying to achieve.
  - Any relevant code snippets or examples.
  - The version of the package you are using.

We encourage asking questions as issues so that others with similar questions can benefit from the answers.


### Reporting Issues/Bugs
- Use the [issue tracker](https://github.com/datasciencecampus/classifai/issues) to report bugs or suggest features.
- Provide as much detail as possible, including steps to reproduce the issue.
#### Providing a (good) description of the bug
When describing the bug it would be beneficial to include:
- Steps on how to reproduce the bug,
- The expected, correct behaviour of the software,
- The actual behaviour of the software,
- How often the bug reproduces


### Suggesting an Enhancement

If you have an idea for a new feature or improvement but do not plan to implement it yourself, you can suggest an enhancement by opening an Issue on our [GitHub Issues page](https://github.com/datasciencecampus/classifai/issues). Please use a clear and descriptive title, and provide details about the proposed enhancement, including its potential benefits and any relevant context. While we welcome all suggestions, submitting an Issue does not guarantee that the feature will be implemented. The maintainers will review and prioritize requests based on project goals and available resources.
### Submitting/Suggesting Code Changes
Whether you want to: suggest a fix to a known bug/issue; want to propose an enhancement; or provide a general beneficial suggestion change to the code base, please follow the below steps to acheive this.

1. Fork the repository.
2. Create a new branch for your changes (`git checkout -b feature-name`).
3. Make your changes and commit them with clear, concise messages.
4. Push your branch to your forked repository.
5. Open a pull request to the main repository.

#### Development Notes

We have standardised on [`uv`](https://docs.astral.sh/uv/) as the package management and project orchestration tool for development on ClassifAI.
Several tools and checks provided to support development on ClassifAI to enforce consistency and security (such as linting, secret scanning and running of unit tests) assume a working installation of `uv`. These tools are made available via `make` commands.
We use `pre-commit` hooks to enforce style and security practices, which can be installed via `make setup-git-hooks` or `make setup-git-hooks-no-docker`.
Using GPG-signed commits is not currently enforced, but is strongly encouraged.

We have additional development patterns around unit testing, which are outlined in the Testing section below.

#### Pull Request CI/CD Formatting
We currently use _conventional commits_ in our GitHub workflows. This means that our CI/CD pipeline validates pull request (PR) titles to ensure they follow specific prefixes. These prefixes help optimize our release pipelines and automate labeling. Below is a table listing the prefixes you should use when opening a pull request from a forked repository:

| Prefix       | Description                                      | Example Usage               | Auto-Label       |
|--------------|--------------------------------------------------|-----------------------------|------------------|
| `fix:`       | Indicates a bug fix in the codebase.             | `fix: resolve null pointer` | `bug`           |
| `feat:`      | Introduces a new feature or functionality.       | `feat: add user login`      | `enhancement`   |
| `docs:`      | Updates or improves documentation.               | `docs: update README`       | `documentation` |
| `test:`      | Adds or modifies tests for existing functionality. | `test: add unit tests`      | `test`          |
| `chore:`     | Miscellaneous tasks like build or dependency updates. | `chore: update dependencies` | `chore`         |
| `refactor:`  | Code changes that neither fix a bug nor add a feature. | `refactor: optimize loop`   | `refactor`      |
| `style:`     | Changes that do not affect the meaning of the code (e.g., formatting). | `style: fix linting issues` | _No label_      |
| `perf:`      | Code changes that improve performance.           | `perf: improve query speed` | `performance`   |
| `ci:`        | Changes to CI/CD configuration files and scripts. | `ci: update GitHub Actions` | `CI/CD`         |
| `revert:`    | Reverts a previous commit.                       | `revert: undo feature`      | `revert`        |

Ensure your PR title starts with one of these prefixes to pass validation. Titles that do not conform will fail the CI/CD pipeline.

We ensure a linear commit history by using squash merges for pull requests. This means that all commits from a feature branch will be combined into a single commit when merged into the main branch. 


### Code Style
- Follow the existing code style and conventions you can view the the ways in which we format, lint and check our code during the development proces. Read through the <b>Development Setup section</b> of the readme.md for information on how to set up correctly.
- You can find a setup guide in the readme.md detailing our linting and formatting strategy (primarly using Ruff).
- Ensure your code is well-documented and includes tests where applicable.


### Testing
We are developing a testing harness of `pytest` unit tests to ensure changes to one area to not silently cause problems in another.
Developers can run these tests locally as they work to confirm their understanding of how their changes impact other areas of the codebase using the following commands;

- `make run-tests-all` : run all unit tests
- `make run-tests-vectorisers` : run unit tests associated with the vectorisers module
- `make run-tests-indexers` : run unit tests associated with the indexers module (WIP)
- `make run-tests-servers` : run unit tests associated with the indexers module (planned)
- `make run-tests-evaluation` : run unit tests associated with the evaluation module (planned)

When marking a PR as ready for review, developers are expected to ensure the unit tests are up-to-date and passing.
If your PR introduces new features, targeted new tests should be written to ensure the behaviour is as-intended.
If your PR introduces changes to existing code, or how aspects of ClassifAI interact with each other, the existing tests should be modified to accurately describe and test the desired behaviour.
The CICD pipeline will run all unit tests on your PR as a check which must pass before it can be merged to `main`.

We recommend opening a PR marked 'Draft' while you continue to work on it to give other developers sight of your progress without running the CICD pipeline.
When you feel it is ready to be merged, you can mark it as ready for review which will trigger the CICD pipeline.
We suggest locally running the tests before this stage to avoid surprises and/or reviewer burden, but we do not currently mandate this via pre-commit checking.


## Getting Help
If you have any questions, feel free to reach out by [opening an issue](https://github.com/datasciencecampus/classifai/issues).

## Release Process (project maintainers only)

The project team decides when to make a new version release. 
The project follows Semantic Versioning principles. To release a new version, the maintainers will:

1. Ensure that all changes intended for the new version are merged into the main branch.
2. Create a new version entry in the `CHANGELOG.md` file.
3. Run the manually dispatched GitHub Action workflow `Release` to validate the changelog and create a new GitHub release.


---

Thank you for contributing! 😊
