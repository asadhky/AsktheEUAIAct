"""The Dockerfiles, `.dockerignore` files and docker-compose.yml are code that makes promises: no secret in an image, no
access log, no model download at run time, one worker, the build arguments agreeing with the settings. Nothing builds
an image in CI, so these tests read the files and check the promises directly."""

import re
from pathlib import Path

import pytest
import yaml

from askact.config import Settings

ROOT = Path(__file__).resolve().parents[2]
BACKEND_DOCKERFILE = ROOT / "backend" / "Dockerfile"
FRONTEND_DOCKERFILE = ROOT / "frontend" / "Dockerfile"
COMPOSE = ROOT / "docker-compose.yml"
SETTINGS = Settings(_env_file=None)


def instructions(path: Path) -> list[tuple[str, str]]:
    """(INSTRUCTION, arguments) for each instruction, with `\\` continuations joined and comments dropped."""
    joined, lines = "", []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.lstrip().startswith("#") and not joined:
            continue
        joined += raw.rstrip("\\").strip() + " " if raw.rstrip().endswith("\\") else raw.strip()
        if raw.rstrip().endswith("\\"):
            continue
        if joined.strip():
            name, _, rest = joined.strip().partition(" ")
            lines.append((name.upper(), rest.strip()))
        joined = ""
    return lines


def index_of(items: list[tuple[str, str]], name: str, containing: str) -> int:
    for i, (instruction, args) in enumerate(items):
        if instruction == name and containing in args:
            return i
    raise AssertionError(f"no {name} instruction containing {containing!r}")


# --- the .dockerignore files are allowlists ---------------------------------------------------------------------

@pytest.mark.parametrize("path", [ROOT / ".dockerignore", ROOT / "frontend" / ".dockerignore"])
def test_a_dockerignore_starts_by_excluding_everything(path):
    rules = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
    assert rules[0] == "*"


@pytest.mark.parametrize("path", [ROOT / ".dockerignore", ROOT / "frontend" / ".dockerignore"])
def test_no_dockerignore_allows_an_env_file_git_or_installed_modules(path):
    allowed = [line.strip()[1:] for line in path.read_text(encoding="utf-8").splitlines() if line.strip().startswith("!")]
    assert allowed, "an allowlist that allows nothing builds nothing"
    for rule in allowed:
        assert not re.search(r"(^|/)\.env|\.git|node_modules|\.venv|\.next|data/index|eval", rule), rule


def test_the_backend_context_allows_only_the_package_and_an_optional_copy_of_the_act():
    allowed = [line.strip()[1:] for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines() if line.startswith("!")]
    assert sorted(allowed) == ["backend/askact", "backend/pyproject.toml", "data/raw/*.html"]


def test_every_file_a_dockerfile_copies_is_allowed_by_its_dockerignore():
    # The frontend build once failed because next.config.mjs was copied but not allowed; this is that mistake, caught.
    for dockerfile, ignore in [(FRONTEND_DOCKERFILE, ROOT / "frontend" / ".dockerignore")]:
        allowed = {line.strip()[1:] for line in ignore.read_text(encoding="utf-8").splitlines() if line.startswith("!")}
        for instruction, args in instructions(dockerfile):
            if instruction != "COPY" or "--from=" in args:
                continue
            *sources, _destination = [token for token in args.split() if not token.startswith("--")]
            for source in sources:
                assert any(source == rule or source.startswith(rule + "/") for rule in allowed), f"{source} is copied but not allowed"


# --- the backend image ----------------------------------------------------------------------------------------------

def test_the_backend_image_copies_no_env_file_and_never_the_whole_context():
    for instruction, args in instructions(BACKEND_DOCKERFILE):
        if instruction in {"COPY", "ADD"}:
            sources = [t for t in args.split() if not t.startswith("--")][:-1]
            assert all(source not in {".", "./", "*"} and ".env" not in source for source in sources), args


def test_the_backend_image_runs_one_worker_with_no_access_log_and_no_proxy_header_rewriting():
    cmd = next(args for instruction, args in instructions(BACKEND_DOCKERFILE) if instruction == "CMD")
    words = yaml.safe_load(cmd)  # exec form is a JSON list
    assert words[0] == "uvicorn"
    assert "--factory" in words and "askact.api:build_app" in words
    assert words[words.index("--workers") + 1] == "1"  # the rate limiter keeps its state in the process
    assert "--no-access-log" in words  # uvicorn's access log records the client address and path
    assert "--no-proxy-headers" in words  # the app decides about X-Forwarded-For itself (TRUSTED_PROXY_HEADER)
    assert words[words.index("--host") + 1] == "0.0.0.0"  # reachable from outside the container


def test_the_models_are_downloaded_before_the_image_goes_offline_and_the_app_runs_as_a_user_that_is_not_root():
    items = instructions(BACKEND_DOCKERFILE)
    ingest = index_of(items, "RUN", "python -m askact.ingest")
    reranker = index_of(items, "RUN", "get_reranker")
    offline = index_of(items, "ENV", "HF_HUB_OFFLINE=1")
    user = index_of(items, "USER", "app")
    cmd = index_of(items, "CMD", "uvicorn")
    assert ingest < offline and reranker < offline  # a download after HF_HUB_OFFLINE=1 would fail the build
    assert offline < user < cmd
    assert not any(instruction == "USER" and args.strip() in {"root", "0"} for instruction, args in items)


def test_torch_comes_from_the_cpu_index_before_the_package_that_depends_on_it():
    items = instructions(BACKEND_DOCKERFILE)
    torch = index_of(items, "RUN", "pip install torch")
    package = index_of(items, "RUN", "[models]")
    assert "download.pytorch.org/whl/cpu" in items[torch][1] and torch < package


def test_the_spend_database_directory_exists_and_belongs_to_the_app_user_so_a_new_volume_is_writable():
    items = instructions(BACKEND_DOCKERFILE)
    prepare = items[index_of(items, "RUN", "mkdir -p /var/lib/askact")][1]
    assert "chown app:app /var/lib/askact" in prepare
    assert SETTINGS.spend_db_path.parent == Path("/var/lib/askact")


def test_the_backend_build_argument_defaults_are_the_settings_defaults():
    # If these drift, the index is built with one model and the app asks for another: /health is 503 forever.
    defaults = {args.split("=", 1)[0]: args.split("=", 1)[1] for instruction, args in instructions(BACKEND_DOCKERFILE) if instruction == "ARG"}
    assert defaults == {
        "EMBEDDING_MODEL": SETTINGS.embedding_model,
        "RERANKER_MODEL": SETTINGS.reranker_model,
        "CHUNK_MAX_CHARS": str(SETTINGS.chunk_max_chars),
    }


def test_the_build_arguments_are_also_the_running_apps_environment():
    env = " ".join(args for instruction, args in instructions(BACKEND_DOCKERFILE) if instruction == "ENV")
    for name in ("EMBEDDING_MODEL", "RERANKER_MODEL", "CHUNK_MAX_CHARS"):
        assert f"{name}=${{{name}}}" in env


# --- docker-compose.yml ---------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def compose() -> dict:
    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))


def test_published_ports_are_bound_to_this_machine_only(compose):
    for name, service in compose["services"].items():
        for port in service["ports"]:
            assert port.startswith("127.0.0.1:"), f"{name} publishes {port} on every network interface"


def test_the_backend_build_arguments_hold_no_secret_and_default_to_the_settings(compose):
    args = compose["services"]["backend"]["build"]["args"]
    assert set(args) == {"EMBEDDING_MODEL", "RERANKER_MODEL", "CHUNK_MAX_CHARS"}  # a build arg is visible in the image history
    assert args["EMBEDDING_MODEL"] == f"${{EMBEDDING_MODEL:-{SETTINGS.embedding_model}}}"
    assert args["RERANKER_MODEL"] == f"${{RERANKER_MODEL:-{SETTINGS.reranker_model}}}"
    assert args["CHUNK_MAX_CHARS"] == f"${{CHUNK_MAX_CHARS:-{SETTINGS.chunk_max_chars}}}"


def test_the_runtime_settings_come_from_an_optional_env_file_so_a_fresh_clone_can_build(compose):
    assert compose["services"]["backend"]["env_file"] == [{"path": ".env", "required": False}]


def test_the_backend_builds_from_the_repository_root_with_its_own_dockerfile(compose):
    build = compose["services"]["backend"]["build"]
    assert (build["context"], build["dockerfile"]) == (".", "backend/Dockerfile")
    assert (ROOT / build["dockerfile"]).exists()


def test_the_spend_ledger_is_on_a_named_volume_at_the_path_the_settings_use(compose):
    mounts = compose["services"]["backend"]["volumes"]
    assert mounts == [f"state:{SETTINGS.spend_db_path.parent}"]
    assert "state" in compose["volumes"]


def test_the_healthcheck_asks_the_health_endpoint_and_has_time_to_load_the_models(compose):
    check = compose["services"]["backend"]["healthcheck"]
    assert "/health" in " ".join(check["test"]) and "curl" not in " ".join(check["test"])  # the image has no curl
    assert int(check["start_period"].rstrip("s")) >= 60


def test_the_ui_waits_for_a_healthy_backend(compose):
    assert compose["services"]["ui"]["depends_on"] == {"backend": {"condition": "service_healthy"}}


def test_the_ui_api_url_is_a_build_argument_defaulting_to_the_local_backend(compose):
    ui = compose["services"]["ui"]
    assert ui["build"]["context"] == "./frontend"
    assert ui["build"]["args"] == {"NEXT_PUBLIC_API_URL": "${NEXT_PUBLIC_API_URL:-http://localhost:8000}"}
    assert "environment" not in ui  # setting it at run time would do nothing: it is inlined at build time


# --- the frontend image ---------------------------------------------------------------------------------------------

def test_the_frontend_image_takes_the_api_url_at_build_time_and_runs_as_node():
    items = instructions(FRONTEND_DOCKERFILE)
    arg = index_of(items, "ARG", "NEXT_PUBLIC_API_URL=http://localhost:8000")
    build = index_of(items, "RUN", "npm run build")
    assert arg < build  # it has to be known when the page is built
    assert not any(instruction == "USER" and args.strip() in {"root", "0"} for instruction, args in items)
    users = [i for i, (instruction, _) in enumerate(items) if instruction == "USER"]
    assert users and items[users[-1]] == ("USER", "node") and users[-1] < index_of(items, "CMD", "server.js")


def test_the_frontend_default_api_url_is_the_one_documented_in_env_example():
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert re.search(r"^NEXT_PUBLIC_API_URL=http://localhost:8000$", example, re.MULTILINE)
