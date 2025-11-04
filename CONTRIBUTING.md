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
### Got a question about the package?
If you have a question about how to use the package, encounter unexpected behavior, or need clarification on something, feel free to [open an issue](#). When asking a question:

- Check the [README file](./README.md), DEMO ipynb [notebooks](./DEMO/), and existing [issues page](https://github.com/datasciencecampus/classifai/issues) to see if your question has already been answered.
- Use a clear and descriptive title for your issue.
- Provide as much context as possible, including:
  - What you are trying to achieve.
  - Any relevant code snippets or examples.
  - The version of the package you are using.

We encourage asking questions as issues so that others with similar questions can benefit from the answers.


### Reporting Issues/Bugs
- Use the [issue tracker](https://github.com/classifAI_package/issues) to report bugs or suggest features.
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


### Code Style
- Follow the existing code style and conventions you can view the the ways in which we format, lint and check our code during the development proces. Read through the <b>Development Setup section</b> of the readme.md for information on how to set up correctly.
- You can find a setup guide in the readme.md detailing our linting and formatting strategy (primarly using Ruff).
- Ensure your code is well-documented and includes tests where applicable.


### Testing
We currently do not have a testing pipeline setup for this repository.


## Getting Help
If you have any questions, feel free to reach out by [opening an issue](https://github.com/datasciencecampus/classifai/issues).

---

Thank you for contributing! 😊
