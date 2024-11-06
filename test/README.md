# tests

This directory contains tests that validate the LLM Observability instrumentation libraries.


## architecture

The test framework we use is [pytest](https://docs.pytest.org/en/stable/). In order to run tests across the different
libraries, the common functionality is abstracted into language-specific HTTP servers that serve the different
features as endpoints (see Dockerfiles and server.{py,js,}). An HTTP client is used in test cases to query the
different servers. Data produced in the servers is routed to a test agent which receives and persists traces and MLObs
requests to be returned back to the test case.

## installation

```bash
pip install -r requirements.txt
```

## format/lint

```bash
ruff format
ruff check
```

## running

```bash
# run all the tests
pytest

# run all the tests in a file
pytest tests/test_sdk.py

# run a specific test
pytest -k test_sdk_

# run tests for a particular set of libraries
TEST_LIBS=nodejs pytest ...
TEST_LIBS=python,nodejs pytest ...
```

## troubleshooting

### docker objects left running

Sometimes if the test framework fails to exit cleanly then some docker containers may be left running. To clean them up:

```bash
docker ps  # list any running containers
docker kill <container_id>
```


```bash
docker network ls  # list any networks

# rm any networks prefixed with llmobs-test
docker network rm $(docker network ls --filter name=llmobs-test -q)
```


### `subprocess.CalledProcessError: Command '['/usr/local/bin/docker', 'network', 'rm', 'llmobs-test-...']' returned non-zero exit status 1.`

This likely means that a container wasn't shutdown during the test and so the network cannot be removed.
