# Publishing and installation

Pybms is packaged as `pybms`, with import name `pybms`, version `0.2.0`, and Python requirement `>=3.10`.
GitHub/source and wheel installs work independently of PyPI. A GitHub push does not create a PyPI package.

## Install from source now

```bash
python -m pip install "git+https://github.com/aymanboufounas/Pybms.git"
```

With optional Excel/Parquet readers:

```bash
python -m pip install "pybms[io] @ git+https://github.com/aymanboufounas/Pybms.git"
```

Or download/clone the repository, then:

```bash
python -m pip install .
```

## Build and validate a release

```bash
python -m pip install ".[dev,io]"
python -m ruff check .
python -m pytest -q
python tools/generate_docs.py
python -m build
python -m twine check dist/*
```

This produces `dist/pybms-0.2.0-py3-none-any.whl` and `dist/pybms-0.2.0.tar.gz`.
Use `python -m pip install dist/pybms-0.2.0-py3-none-any.whl` to install the wheel.

## Enable the short PyPI command

The owner must have a PyPI account and register a Trusted Publisher (or a pending publisher for a new project).
First verify that the PyPI project name is still available; names can be registered by other users.

Use these exact publisher settings:

| Setting | Value |
|---|---|
| PyPI project name | `pybms` |
| GitHub owner | `aymanboufounas` |
| GitHub repository | `Pybms` |
| Workflow filename | `publish.yml` |
| GitHub environment | `pypi` |

Register at [PyPI publishing settings](https://pypi.org/manage/account/publishing/).
See [PyPI pending publishers](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/) for a new project.
Create the GitHub `pypi` environment in repository settings to match the publisher configuration.
Then publish a GitHub release tagged `v0.2.0`, or run **Publish Pybms to PyPI** manually on `main`.
The workflow runs tests, builds distributions, verifies metadata, and publishes using OIDC without a stored API token.
It requires PyPI's publisher configuration; code alone cannot establish ownership of a PyPI project.

After a successful upload, users can run:

```bash
python -m pip install pybms
python -m pip install "pybms[io]"
```

For subsequent releases, update the version in `pyproject.toml` and `pybms/__init__.py`, update the changelog, and publish a matching new version tag.
PyPI rejects uploading the same filename/version twice. Do not overwrite an existing release.

Reference: [PyPA packaging tutorial](https://packaging.python.org/en/latest/tutorials/packaging-projects/) and [PyPI Trusted Publishers](https://docs.pypi.org/trusted-publishers/).
