import hashlib
from pathlib import Path

import httpx
import pytest

from ragferee.download import Downloader, DownloadError, RemoteFile

CONTENT = b"%PDF-fake laws of the game"
FILE = RemoteFile(
    url="https://example.test/laws.pdf",
    filename="laws.pdf",
    sha256=hashlib.sha256(CONTENT).hexdigest(),
)


def make_downloader(destination: Path, status: int = 200) -> tuple[Downloader, list[str]]:
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(status, content=CONTENT)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return Downloader(client, destination), requests


def test_download_saves_verified_file(tmp_path: Path) -> None:
    downloader, _ = make_downloader(tmp_path)

    path = downloader.download(FILE)

    assert path == tmp_path / "laws.pdf"
    assert path.read_bytes() == CONTENT
    assert list(tmp_path.iterdir()) == [path]


def test_download_skips_existing_file(tmp_path: Path) -> None:
    downloader, requests = make_downloader(tmp_path)
    (tmp_path / "laws.pdf").write_bytes(b"already here")

    path = downloader.download(FILE)

    assert path.read_bytes() == b"already here"
    assert requests == []


def test_download_force_replaces_existing_file(tmp_path: Path) -> None:
    downloader, requests = make_downloader(tmp_path)
    (tmp_path / "laws.pdf").write_bytes(b"old")

    path = downloader.download(FILE, force=True)

    assert path.read_bytes() == CONTENT
    assert len(requests) == 1


def test_download_rejects_checksum_mismatch(tmp_path: Path) -> None:
    downloader, _ = make_downloader(tmp_path)
    wrong = FILE.model_copy(update={"sha256": "0" * 64})

    with pytest.raises(DownloadError, match="Checksum mismatch"):
        downloader.download(wrong)

    assert list(tmp_path.iterdir()) == []


def test_download_wraps_http_errors(tmp_path: Path) -> None:
    downloader, _ = make_downloader(tmp_path, status=404)

    with pytest.raises(DownloadError, match="Could not download"):
        downloader.download(FILE)

    assert list(tmp_path.iterdir()) == []
