# llm-bot-playground

Inspired by https://github.com/kentaro/research_assistant


## How to develop
```shell
$ python -m venv .venv # to inform poetry of the .python-version
$ poetry install

$ poetry shell
$ OPENAI_API_KEY=xxx streamlit app.py

$ poetry run python -m flake8 **/*.py # lint
$ poetry run python -m black **/*.py # format
```
