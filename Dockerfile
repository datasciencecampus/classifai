# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.10-buster

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

COPY . ./

RUN python -m pip install --upgrade pip
RUN python -m pip install -e ".[dev]" --no-cache-dir

# Run the application.
CMD python fast-api/main_api.py
