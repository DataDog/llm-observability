import dataclasses
import platform
import shutil
import subprocess
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Union


def _docker_exec() -> str:
    return shutil.which("docker")


def running() -> bool:
    try:
        subprocess.run(
            [_docker_exec(), "info"], check=True, text=True, capture_output=True
        )
        return True
    except subprocess.CalledProcessError:
        return False


def get_platform() -> Union[Literal["amd64"], Literal["arm64"]]:
    p = platform.platform()
    if "x86_64" in p:
        return "amd64"
    if "aarch64" in p or "arm64" in p:
        return "arm64"
    raise ValueError(f"Unsupported architecture for '{p}'")


def docker_platform():
    a = get_platform()
    if a == "amd64":
        return "linux/amd64"
    return "linux/arm64/v8"


@dataclasses.dataclass
class DockerBuildError(Exception):
    image: str
    stdout: Optional[str]
    stderr: Optional[str]

    def __str__(self):
        return f"Error building image '{self.image}': {self.stderr}"


def docker_build(
    tag,
    dockerfile,
    context,
    platform=None,
    default_volumes=None,
    default_network=None,
    default_detach=None,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
):
    if platform is None:
        platform = docker_platform()
    cmd = [
        shutil.which("docker"),
        "build",
        f"--platform={platform}",
        "--progress=plain",
        f"--tag={tag}",
        f"--file={dockerfile}",
        context,
    ]
    p = subprocess.run(cmd, stdout=stdout, stderr=stderr)
    if p.returncode != 0:
        raise DockerBuildError(
            image=tag,
            stdout=p.stdout.decode(errors="replace") if p.stdout else None,
            stderr=p.stderr.decode(errors="replace") if p.stderr else None,
        )
    return DockerImage(
        image=tag,
        default_volumes=default_volumes,
        default_network=default_network,
        default_detach=default_detach,
    )


class DockerContainer:
    def __init__(self, container_id):
        self.container_id = container_id

    def kill(self):
        subprocess.run(
            [shutil.which("docker"), "kill", "--signal=SIGKILL", self.container_id],
            check=True,
            timeout=1,
        )

    def wait(self, timeout=None):
        subprocess.run(
            [shutil.which("docker"), "wait", self.container_id],
            check=True,
            timeout=timeout,
        )

    def remove(self, force=False):
        subprocess.run([shutil.which("docker"), "kill", self.container_id], check=True)

    def is_running(self):
        try:
            result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", self.container_id],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            return result.stdout.strip() == "true"
        except subprocess.CalledProcessError as e:
            print(f"Error checking container status: {e.stderr}")
            return False

    def logs(self, stdout=True, stderr=False) -> str:
        assert stdout ^ stderr
        p = subprocess.run(
            [shutil.which("docker"), "logs", self.container_id],
            capture_output=True,
            check=True,
            text=True,
        )
        if stdout:
            return p.stdout
        return p.stderr


def docker_run(
    image: str,
    environment: Union[Dict[str, str], List[str]],
    volumes: List[str],
    ports: Dict[str, str],
    network: str,
    detach: bool,
    name: Optional[str] = None,
    platform: Optional[bool] = None,
    command: Optional[List[str]] = None,
) -> Union[Tuple[bytes, bytes], DockerContainer]:
    if platform is None:
        platform = docker_platform()
    cmd = [
        shutil.which("docker"),
        "run",
        "--rm",
        f"--platform={platform}",
    ]
    if network:
        cmd.extend(["--network", f"{network}"])
    if detach:
        cmd.append("--detach")
    if name:
        cmd.extend(["--name", name])
    for volume in volumes:
        cmd.extend(["--volume", f"{volume}"])
    for host, container in ports.items():
        cmd.extend(["--publish", f"{host}:{container}"])
    if isinstance(environment, list):
        for env in environment:
            cmd.extend(["--env", env])
    else:
        for key, value in environment.items():
            cmd.extend(["--env", f"{key}={value}"])
    cmd.append(image)
    if command:
        cmd.extend(command)

    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        # text=True
    )
    if detach:
        return DockerContainer(container_id=p.stdout.decode().strip())
    return p.stdout, p.stderr


@dataclasses.dataclass
class DockerImage:
    image: str
    default_detach: bool
    default_network: str = ""
    default_volumes: List[str] = dataclasses.field(default_factory=list)

    def run(
        self,
        cmd: List[str] = [],
        env: Dict[str, str] = {},
        volumes: Optional[List[str]] = None,
        ports: Dict[str, str] = {},
        network: Optional[str] = None,
        detach: Optional[bool] = None,
        platform: Optional[str] = "",
    ):
        if network is None:
            network = self.default_network
        if volumes is None:
            volumes = self.default_volumes
        if detach is None:
            detach = self.default_detach
        if platform is None:
            platform = docker_platform()
        return docker_run(
            image=self.image,
            command=cmd,
            environment=env,
            volumes=volumes,
            platform=platform,
            ports=ports,
            network=network,
            detach=detach,
        )
