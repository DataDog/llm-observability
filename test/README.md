# tests

This directory contains tests that validate our instrumentation libraries.


## architecture

The test framework we use is [pytest](https://docs.pytest.org/en/stable/). In order to run tests across the different
libraries, the common functionality is abstracted into language-specific HTTP servers that serve the different
features as endpoints (see Dockerfiles and server.{py,js,}). An HTTP client is used in test cases to query the
different servers. Data produced in the servers is routed to a test agent which receives and persists traces and MLObs
requests to be returned back to the test case.
