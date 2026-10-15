"""Download source documents with checksum verification."""

import hashlib
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict


class RemoteFile(BaseModel):
    """A file to download, with the checksum it must match."""

    model_config = ConfigDict(frozen=True)

    url: str
    filename: str
    sha256: str


class DownloadError(Exception):
    """Raised when a download fails or its checksum does not match."""


class Downloader:
    """Downloads files into a directory and verifies their checksums."""

    def __init__(self, client: httpx.Client, destination: Path) -> None:
        self.client = client
        self.destination = destination

    def download(self, file: RemoteFile, *, force: bool = False) -> Path:
        """Download a file unless it already exists, and return its path."""
        target = self.destination / file.filename
        if target.exists() and not force:
            return target

        self.destination.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".part")
        try:
            actual = self._fetch(file, part)
            self._verify(file, actual)
            part.replace(target)
        except httpx.HTTPError as error:
            raise DownloadError(f"Could not download {file.url}: {error}") from error
        finally:
            part.unlink(missing_ok=True)
        return target

    def _fetch(self, file: RemoteFile, part: Path) -> str:
        digest = hashlib.sha256()
        with self.client.stream("GET", file.url) as response:
            response.raise_for_status()
            with part.open("wb") as out:
                for chunk in response.iter_bytes():
                    out.write(chunk)
                    digest.update(chunk)
        return digest.hexdigest()

    def _verify(self, file: RemoteFile, actual: str) -> None:
        expected = file.sha256.lower()
        if actual != expected:
            raise DownloadError(
                f"Checksum mismatch for {file.url}: expected {expected}, got {actual}"
            )


LAWS_OF_THE_GAME = RemoteFile(
    url="https://downloads.theifab.com/downloads/laws-of-the-game-202627-single-pages?l=en",
    filename="laws-of-the-game-2026-27.pdf",
    sha256="89398520b353c6d995a1bb2557c97dc3c5aa1712c1d3009589f580a5a23c6a7d",
)
