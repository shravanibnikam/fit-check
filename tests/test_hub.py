import io
import json
import urllib.error
import urllib.parse
import urllib.request

import pytest

from fitcheck import hub
from fitcheck.safetensors_header import count_params

CONFIG = {"model_type": "llama", "hidden_size": 4}


class FakeOpener:
    """Stands in for the urllib opener: serves files from a dict, honoring Range."""

    def __init__(self, files, honor_range=True, error_code=None):
        self.files = files
        self.honor_range = honor_range
        self.error_code = error_code
        self.requests = []

    def open(self, req, timeout):
        self.requests.append(req)
        name = urllib.parse.unquote(req.full_url.rsplit("/", 1)[1])
        if self.error_code or name not in self.files:
            code = self.error_code or 404
            raise urllib.error.HTTPError(req.full_url, code, "error", {}, None)
        data, status = self.files[name], 200
        rng = req.get_header("Range")
        if rng and self.honor_range:
            start, end = map(int, rng.removeprefix("bytes=").split("-"))
            data, status = data[start : end + 1], 206
        resp = io.BytesIO(data)
        resp.status = status
        return resp


def sharded_files(make_safetensors):
    index = {"weight_map": {"a": "s-1.safetensors", "b": "s-2.safetensors"}}
    return {
        "config.json": json.dumps(CONFIG).encode(),
        "model.safetensors.index.json": json.dumps(index).encode(),
        "s-1.safetensors": make_safetensors({"a": [4, 4]}),
        "s-2.safetensors": make_safetensors({"b": [4, 2]}),
    }


def write_folder(folder, files):
    for name, data in files.items():
        (folder / name).write_bytes(data)
    return str(folder)


def test_local_single_file(tmp_path, make_safetensors):
    folder = write_folder(
        tmp_path,
        {
            "config.json": json.dumps(CONFIG).encode(),
            "model.safetensors": make_safetensors({"w": [3, 5]}),
        },
    )
    config, tensors = hub.load(folder)
    assert config == CONFIG
    assert count_params(tensors) == 15


def test_local_sharded(tmp_path, make_safetensors):
    folder = write_folder(tmp_path, sharded_files(make_safetensors))
    _, tensors = hub.load(folder)
    assert set(tensors) == {"a", "b"}
    assert count_params(tensors) == 16 + 8


def test_local_without_safetensors(tmp_path):
    folder = write_folder(tmp_path, {"config.json": b"{}"})
    with pytest.raises(FileNotFoundError, match="only safetensors"):
        hub.load(folder)


def test_rejects_shard_outside_repo(tmp_path, make_safetensors):
    files = sharded_files(make_safetensors)
    index = {"weight_map": {"a": "../secret.safetensors"}}
    files["model.safetensors.index.json"] = json.dumps(index).encode()
    with pytest.raises(ValueError, match="shard file name"):
        hub.load(write_folder(tmp_path, files))


def test_hub_sharded_uses_range_requests(monkeypatch, make_safetensors):
    opener = FakeOpener(sharded_files(make_safetensors))
    monkeypatch.setattr(hub, "_OPENER", opener)
    config, tensors = hub.load("org/model", revision="refs/pr/1", token="secret")
    assert config == CONFIG
    assert count_params(tensors) == 24
    first = opener.requests[0]
    assert (
        first.full_url == f"{hub.HUB_URL}/org/model/resolve/refs%2Fpr%2F1/config.json"
    )
    assert first.get_header("Authorization") == "Bearer secret"
    ranges = [r.get_header("Range") for r in opener.requests if r.get_header("Range")]
    assert ranges[0] == "bytes=0-7"
    assert all(r.get_header("Range") for r in opener.requests if "s-" in r.full_url)


def test_hub_refuses_full_download_when_range_ignored(monkeypatch, make_safetensors):
    files = {
        "config.json": b"{}",
        "model.safetensors": make_safetensors({"w": [1]}),
    }
    monkeypatch.setattr(hub, "_OPENER", FakeOpener(files, honor_range=False))
    with pytest.raises(RuntimeError, match="ignored the range"):
        hub.load("org/model")


def test_hub_access_denied_mentions_token(monkeypatch):
    monkeypatch.setattr(hub, "_OPENER", FakeOpener({}, error_code=401))
    with pytest.raises(PermissionError, match="HF_TOKEN"):
        hub.load("meta-llama/Llama-3.1-8B")


@pytest.mark.parametrize(
    ("target", "keeps_token"),
    [
        ("/api/resolve-cache/models/org/model/abc/config.json", True),
        ("https://cdn.example.com/signed?sig=1", False),
    ],
)
def test_token_only_follows_same_host_redirects(target, keeps_token):
    req = urllib.request.Request(f"{hub.HUB_URL}/org/model/resolve/main/x")
    req.add_header("Authorization", "Bearer secret")
    req.add_header("Range", "bytes=0-7")
    newurl = urllib.parse.urljoin(req.full_url, target)
    handler = hub._DropAuthOnCrossHostRedirect()
    new = handler.redirect_request(req, None, 302, "Found", {}, newurl)
    assert new.has_header("Authorization") == keeps_token
    assert new.get_header("Range") == "bytes=0-7"


@pytest.mark.network
def test_qwen_from_hub():
    # Pinned so the expected count can't drift with new uploads.
    config, tensors = hub.load(
        "Qwen/Qwen2.5-0.5B", revision="060db6499f32faf8b98477b0a26969ef7d8b9987"
    )
    assert config["num_hidden_layers"] == 24
    # Derived from config.json shapes, see docs/memory-model.md. The output head
    # shares the embedding matrix (tie_word_embeddings), so it is stored once.
    assert count_params(tensors) == 494_032_768
