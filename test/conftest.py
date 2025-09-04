import base64
import json
import os
from packaging.version import Version
import requests
import shutil
import socket
import subprocess
import time
import uuid

import pytest
import msgpack
import gzip

from test import docker
from test.client import InstrumentationClient
from ddapm_test_agent.client import TestAgentClient

from typing import Any
from typing import Dict
from typing import Tuple


MLOBS_NAMESPACE = "mlobs"
TEST_LIBS = os.environ.get("TEST_LIBS", "python,nodejs,java").split(",")
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
IGNORE_PARAMS_FOR_TEST_NAME = (
    "test_lang",
    "test_agent_connectivity_mode",
    "test_client_ml_app",
)


def pytest_report_header():
    return [
        "sdks: " + ", ".join(TEST_LIBS),
    ]


@pytest.fixture(scope="session")
def log_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("logs")


@pytest.fixture(scope="session")
def docker_network_name():
    return f"llmobs-test-{str(uuid.uuid4())[0:8]}"


@pytest.fixture(scope="session")
def docker_network(docker_network_name):
    docker = shutil.which("docker")
    subprocess.run([docker, "network", "create", docker_network_name], check=True)
    yield docker_network_name
    subprocess.run([docker, "network", "rm", docker_network_name], check=True)


@pytest.fixture(scope="session", autouse=True)
def ensure_docker_running() -> None:
    if not docker.running():
        pytest.fail("Docker is not running. Please start Docker and re-run the tests.")


@pytest.fixture(params=TEST_LIBS)
def test_lang(request) -> str:
    assert request.param in [
        "python",
        "nodejs",
        "java",
    ], f"Invalid test language '{request.param}' provided"
    return request.param


def _find_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))  # Bind to a free port provided by the host.
    port = s.getsockname()[1]  # Get the port number assigned.
    s.close()  # Release the socket.
    # assume nothing else grabs the port 🤞
    return port


@pytest.fixture(scope="session")
def testagent_port():
    return _find_port()


@pytest.fixture(scope="session")
def testagent_docker_name():
    return "mlobs-test-agent"


class LLMObsTestAgentClient(TestAgentClient):
    """Extend TestAgentClient to provide additional functionality for LLMObs."""

    def wait_for_llmobs_requests(self, num, num_tries=10, delay=0.2):
        """Wait for `num` llmobs requests to be received from the test agent."""
        num_received = 0
        attempts = 0
        while num_received < num:
            if attempts >= num_tries:
                break
            reqs = self.llmobs_requests()
            num_received = len(reqs)
            if num_received < num:
                time.sleep(delay)
            attempts += 1
        if num_received < num:
            raise AssertionError(
                f"Expected {num} llmobs requests, received {num_received}"
            )
        return reqs

    def wait_for_llmobs_evaluations_requests(self, num, num_tries=10, delay=0.2):
        """Wait for `num` llmobs evaluations requests to be received from the test agent."""
        num_received = 0
        attempts = 0
        while num_received < num:
            if attempts >= num_tries:
                break
            reqs = self.llmobs_evaluations_requests()
            num_received = len(reqs)
            if num_received < num:
                time.sleep(delay)
            attempts += 1
        if num_received < num:
            raise AssertionError(
                f"Expected {num} llmobs evaluations requests, received {num_received}"
            )
        return reqs

    def wait_for_telemetry_metric(
        self,
        metric_name: str,
        clear: bool = False,
        wait_loops: int = 200,
    ):
        """Wait for and return the given telemetry metric from the test agent."""
        for i in range(wait_loops):
            try:
                events = self.telemetry(clear=False)
            except Exception:
                pass
            else:
                for event in events:
                    if event["request_type"] not in (
                        "generate-metrics",
                        "distributions",
                    ):
                        continue
                    if event["payload"]["namespace"] != MLOBS_NAMESPACE:
                        continue
                    for metric in event["payload"]["series"]:
                        if metric["metric"] == metric_name:
                            if clear:
                                self.clear()
                            return event, metric
                if clear:
                    self.clear()
            time.sleep(0.01)
        raise AssertionError(
            "Telemetry event %s.%s not found" % (MLOBS_NAMESPACE, metric_name)
        )

    def llmobs_requests(self):
        reqs = [
            r
            for r in self.requests()
            if r["url"].endswith("/evp_proxy/v2/api/v2/llmobs")
        ]

        events = []
        for r in reqs:
            decoded_body = base64.b64decode(r["body"])
            try:
                events.append(json.loads(decoded_body))
            except UnicodeDecodeError:
                decompressed = gzip.decompress(decoded_body)
                unpacker = msgpack.Unpacker()
                unpacker.feed(decompressed)
                events.extend([event for event in unpacker])
        return events

    def llmobs_evaluations_requests(self):
        reqs = [
            r
            for r in self.requests()
            if r["url"].endswith("/evp_proxy/v2/api/intake/llm-obs/v1/eval-metric")
            or r["url"].endswith("/evp_proxy/v2/api/intake/llm-obs/v2/eval-metric")
        ]
        return [json.loads(base64.b64decode(r["body"])) for r in reqs]


@pytest.fixture(params=["tcp"])
def test_agent_connectivity_mode(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture
def test_agent_connectivity_env_and_volumes(
    testagent_docker_name,
    testagent_port,
    test_agent_socket_path,
    test_agent_connectivity_mode,
) -> Tuple[Any, Any]:
    if test_agent_connectivity_mode == "tcp":
        env = {"DD_TRACE_AGENT_URL": f"http://{testagent_docker_name}:{testagent_port}"}
        volumes = []
    elif test_agent_connectivity_mode == "uds":
        env = {"DD_TRACE_AGENT_URL": "unix:///var/run/datadog/apm.socket"}
        volumes = [f"{test_agent_socket_path}/:/var/run/datadog/"]
    elif test_agent_connectivity_mode == "api":
        env = {
            "DD_TRACE_AGENT_URL": f"http://{testagent_docker_name}:{testagent_port}",  # use for the rest of the tracer products to connect to
            "DD_LLMOBS_AGENTLESS_ENABLED": "true",  # force LLMObs to talk to DD intake
            "DD_API_KEY": os.environ.get("DD_API_KEY", "test-key"),
        }
        volumes = []
    return env, volumes


@pytest.fixture(scope="session")
def test_agent_socket_path():
    socket_path = "/tmp/run/datadog"
    os.makedirs(socket_path, exist_ok=True)
    yield socket_path
    shutil.rmtree(socket_path)


@pytest.fixture(scope="session")
def _test_agent(
    docker_network, testagent_docker_name, testagent_port, test_agent_socket_path
):
    """
    Run a test agent for the duration of the test session.

    The test agent listens on both a TCP port and a UDS socket so both can be tested.

    A client is returned to interact with the test agent.
    """
    docker_args = dict(
        image="ghcr.io/datadog/dd-apm-test-agent/ddapm-test-agent:v1.31.0",
        environment={
            "PORT": testagent_port,
            "DD_APM_NON_LOCAL_TRAFFIC": "true",
            "DD_APM_RECEIVER_SOCKET": "/var/run/datadog/apm.socket",
            "VCR_CASSETTES_DIRECTORY": "/cassettes",
        },
        ports={testagent_port: testagent_port},
        name=testagent_docker_name,
        network=docker_network,
        volumes=[
            f"{test_agent_socket_path}/:/var/run/datadog/",
            f"{CUR_DIR}/cassettes:/cassettes",
        ],
    )

    c = docker.docker_run(**docker_args, detach=True)
    try:
        agent = LLMObsTestAgentClient(base_url=f"http://localhost:{testagent_port}")
        try:
            # Wait for the agent to start
            agent.wait_to_start(num_tries=10, delay=0.1)
        except AssertionError as e:
            # If the container did start but the agent isn't working then get the logs and reraise.
            if c.is_running():
                raise ConnectionError(
                    f"test agent container started but agent not responsive:\n\n{e}"
                ) from None

            # If the container didn't start then likely there was a bad test agent config in which case
            # we can try to run the container synchronously to get the logs.
            stdout, stderr = docker.docker_run(**docker_args, detach=False)
            raise ConnectionError(
                f"test agent container failed to start, container stderr:\n\n{stderr.decode()}"
            ) from None

        yield agent
    finally:
        if c.is_running():
            c.kill()


@pytest.fixture(scope="function")
def test_agent(_test_agent):
    """
    We run the test agent once for the entirety of the test suite. So in order to not have conflicts between test cases
    the test agent state is cleared between runs.
    """
    _test_agent.clear()
    yield _test_agent
    _test_agent.clear()


@pytest.fixture(autouse=True)
def with_vcr_test_name(request: pytest.FixtureRequest, testagent_port):
    test_name: str = request.node.originalname

    for param in request.node.callspec.params:
        if param not in IGNORE_PARAMS_FOR_TEST_NAME:
            param_value = request.node.callspec.params[param]
            test_name += f"_{param}_{param_value}"

    testagent_url = f"http://127.0.0.1:{testagent_port}"  # this is exposed outside of the docker network
    with requests.post(
        f"{testagent_url}/vcr/test/start", json={"test_name": test_name}, timeout=5
    ) as resp:
        resp.raise_for_status()
        yield
    requests.post(f"{testagent_url}/vcr/test/stop", timeout=5)


@pytest.fixture
def test_lang_specific_tracer_config(test_lang) -> Dict[str, str]:
    if test_lang == "python":
        return {"DD_PATCH_MODULES": "fastapi:false"}
    elif test_lang == "nodejs":
        return {
            "DD_TRACE_EXPRESS_ENABLED": "false",
            "DD_TRACE_HTTP_ENABLED": "false",
            "DD_TRACE_DNS_ENABLED": "false",
            "DD_TRACE_NET_ENABLED": "false",
            "DD_TRACE_FETCH_ENABLED": "false",
        }
    elif test_lang == "java":
        return {
            "DD_INTEGRATION_OKHTTP_ENABLED": "false",
            "DD_INTEGRATION_JETTY_ENABLED": "false",
        }


@pytest.fixture
def lang_local_dev_support(test_lang):
    env = {}
    volumes = []

    if test_lang == "python" and os.path.exists(f"{CUR_DIR}/dd-trace-py/"):
        env["PYTHONPATH"] = "/dd-trace-py/ddtrace/bootstrap:/dd-trace-py"
        volumes.append(f"{CUR_DIR}/dd-trace-py/:/dd-trace-py")

        # assume that if the native code is there we're good
        # this is totally brittle and not good but i haven't found a good way to cache the build
        if not os.path.exists(
            f"{CUR_DIR}/dd-trace-py/ddtrace/internal/_threads.cpython-312-aarch64-linux-gnu.so"
        ):
            # Have to build the linux wheels
            docker.docker_run(
                image="ghcr.io/datadog/dd-trace-py/testrunner:latest",
                detach=False,
                command=[
                    "bash",
                    "-c",
                    "pyenv global 3.12 && python -m pip install -e /dd-trace-py",
                ],
                network="host",
                volumes=volumes,
                environment={},
                ports={},
            )
    elif test_lang == "nodejs" and os.path.exists(f"{CUR_DIR}/dd-trace-js/"):
        volumes.append(f"{CUR_DIR}/dd-trace-js/:/dd-trace-js")

    # mount local directory
    volumes.append("./:/app")

    return env, volumes


@pytest.fixture(params=["test-ml-app"])
def test_client_ml_app(request: pytest.FixtureRequest) -> str:
    """ML application to set on the test client"""
    return request.param


@pytest.fixture
def test_client(
    request,
    docker_network,
    test_lang_specific_tracer_config,
    test_agent_connectivity_env_and_volumes,
    lang_local_dev_support,
    test_lang,
    testagent_docker_name,
    testagent_port,
    test_client_ml_app,
):
    # skip immediately if the test is marked as unsupported
    lib_support = getattr(request.node.function, "library_support", [])
    for lang, version, reason in lib_support:
        if lang == test_lang:
            if version == "unsupported":
                pytest.skip(f"Test does not support {test_lang}, reason: {reason}")

    docker.docker_build(
        f"llmobs-test-server-{test_lang}",
        dockerfile=os.path.join(CUR_DIR, f"Dockerfile.{test_lang}"),
        context=CUR_DIR,
    )

    local_port = str(_find_port())
    internal_port = "8080"
    dd_trace_agent_url = f"http://{testagent_docker_name}:{testagent_port}"

    server_env = {
        # test client specific config
        "PORT": internal_port,
        # datadog
        "DD_TRACE_AGENT_URL": dd_trace_agent_url,
        "DD_TRACE_DEBUG": "true",
        "DD_LLMOBS_ENABLED": True,  # TODO: make this configurable?
        "DD_SERVICE": "test-service",  # TODO: make this configurable?
        # third party client configuration
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
        "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY"),
        "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY"),
        "GOOGLE_APPLICATION_CREDENTIALS": os.path.join(
            "/app", os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
        ),
        "PROXY_LLM_SERVER_URL": f"{dd_trace_agent_url}/vcr",
    }

    if test_client_ml_app is not None:
        server_env["DD_LLMOBS_ML_APP"] = test_client_ml_app

    volumes = []

    server_env.update(test_lang_specific_tracer_config)

    dev_env, dev_vols = lang_local_dev_support
    server_env.update(dev_env)
    volumes.extend(dev_vols)

    connectivity_env, connectivity_vols = test_agent_connectivity_env_and_volumes
    server_env.update(connectivity_env)
    volumes.extend(connectivity_vols)

    server_kwargs = dict(
        image=f"llmobs-test-server-{test_lang}",
        ports={local_port: internal_port},
        network=docker_network,
        environment=server_env,
        volumes=volumes,
    )
    container = docker.docker_run(**server_kwargs, detach=True)

    try:
        c = InstrumentationClient(f"http://0.0.0.0:{local_port}", test_lang=test_lang)
        server_info = c.wait_to_start()

        # Confirm that the test case is compatible with the server
        for lang, version, reason in lib_support:
            if lang == test_lang:
                server_version = Version(server_info["version"])
                min_version = Version(version)

                if server_version.is_prerelease:
                    server_version = Version(server_version.base_version)

                if server_version < min_version:
                    pytest.skip(
                        f"Test does not support {test_lang} {server_info['version']}, min version is '{version}'"
                    )
        yield c
        print(container.logs(stderr=True, stdout=False))
        print(container.logs(stdout=True))
    except AssertionError:
        if container.is_running():
            print(container.logs(stderr=True, stdout=False))
            print(container.logs(stdout=True))
            raise AssertionError("Server did not start, check the logs") from None
        else:
            # If the container didn't start then likely there was a server bug
            # we can try to run the container synchronously to get the logs.
            stdout, stderr = docker.docker_run(**server_kwargs, detach=False)
            raise ConnectionError(
                f"test server container failed to start, container stderr:\n\n{stderr.decode()}"
            ) from None
    finally:
        if container.is_running():
            container.kill()
