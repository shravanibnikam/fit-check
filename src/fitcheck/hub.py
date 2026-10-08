"""Load config.json and safetensors headers from the Hugging Face Hub or a local folder.

This is the only module that touches the network, so the rest of the package
stays pure Python and can run under Pyodide. Weights are never downloaded: for
each safetensors file we fetch the 8-byte length prefix, then the JSON header,
with HTTP range requests (RFC 9110, section 14).
"""

import json
import os
import urllib.error
import urllib.parse
import urllib.request

from fitcheck.safetensors_header import PREFIX_BYTES, header_length, parse_header

HUB_URL = "https://huggingface.co"
TIMEOUT_S = 30

# transformers looks for the single file first, then the shard index
# (_get_resolved_checkpoint_files in transformers/modeling_utils.py).
SINGLE_FILE = "model.safetensors"
INDEX_FILE = "model.safetensors.index.json"


def load(
    model: str, revision: str = "main", token: str | None = None
) -> tuple[dict, dict[str, dict]]:
    """Return (config, tensors) for a Hub model id or a local folder.

    `tensors` maps every tensor name to its header entry (dtype, shape).
    `token` defaults to the HF_TOKEN environment variable, needed for gated repos.
    """
    if os.path.isdir(model):
        read = _local_reader(model)
    else:
        read = _hub_reader(model, revision, token or os.environ.get("HF_TOKEN"))
    config = json.loads(read("config.json"))
    return config, _read_tensors(read)


def _read_tensors(read) -> dict[str, dict]:
    try:
        return _read_header(read, SINGLE_FILE)
    except FileNotFoundError:
        pass
    try:
        index = json.loads(read(INDEX_FILE))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"found neither {SINGLE_FILE} nor {INDEX_FILE}; "
            "only safetensors checkpoints are supported"
        ) from None
    tensors = {}
    for shard in sorted(set(index["weight_map"].values())):
        # Shard names come from the repo, so keep them from escaping it.
        if os.path.basename(shard) != shard or shard in ("", ".", ".."):
            raise ValueError(f"unexpected shard file name {shard!r}")
        tensors.update(_read_header(read, shard))
    return tensors


def _read_header(read, name: str) -> dict[str, dict]:
    n = header_length(read(name, 0, PREFIX_BYTES))
    return parse_header(read(name, PREFIX_BYTES, n))


def _local_reader(folder: str):
    def read(name: str, start: int = 0, size: int = -1) -> bytes:
        with open(os.path.join(folder, name), "rb") as f:
            f.seek(start)
            return f.read(size)

    return read


class _DropAuthOnCrossHostRedirect(urllib.request.HTTPRedirectHandler):
    """Keep the token on same-host redirects, drop it when leaving the host.

    The Hub answers /resolve/ for large files with a redirect to a CDN host
    holding a signed URL. urllib copies every header to the redirect target,
    so without this the HF token would go to the CDN too. requests strips it
    the same way (Session.should_strip_auth in requests/sessions.py).
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and _host(newurl) != _host(req.full_url):
            new.remove_header("Authorization")
        return new


def _host(url: str) -> str | None:
    return urllib.parse.urlsplit(url).hostname


_OPENER = urllib.request.build_opener(_DropAuthOnCrossHostRedirect)


def _hub_reader(model: str, revision: str, token: str | None):
    # Same URL shape as hf_hub_url in huggingface_hub/file_download.py.
    base = f"{HUB_URL}/{model}/resolve/{urllib.parse.quote(revision, safe='')}/"

    def read(name: str, start: int = 0, size: int = -1) -> bytes:
        url = base + urllib.parse.quote(name)
        req = urllib.request.Request(url)
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        if size >= 0:
            req.add_header("Range", f"bytes={start}-{start + size - 1}")  # inclusive
        try:
            with _OPENER.open(req, timeout=TIMEOUT_S) as resp:
                if size >= 0 and resp.status != 206:
                    # The server ignored the range and would send the whole file,
                    # gigabytes for a weights shard. Stop before reading the body.
                    raise RuntimeError(f"{url}: server ignored the range request")
                return resp.read(size)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise FileNotFoundError(url) from None
            if e.code in (401, 403):
                # The Hub answers 401 for missing repos too, so as not to leak private ones.
                raise PermissionError(
                    f"{model}: access denied ({e.code}). The repo may be gated, "
                    "private or misspelled. For gated repos, accept the license on "
                    "the Hub and set HF_TOKEN."
                ) from None
            raise

    return read
