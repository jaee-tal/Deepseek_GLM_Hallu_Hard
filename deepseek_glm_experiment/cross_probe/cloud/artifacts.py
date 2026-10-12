"""Small artifact-store abstraction with local and Google Cloud Storage backends."""
from __future__ import annotations

import hashlib
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from urllib.parse import urlparse


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


class ArtifactStore(ABC):
    """Object operations used by checkpoint manifests."""

    @classmethod
    def from_uri(cls, uri: str) -> "ArtifactStore":
        parsed = urlparse(uri)
        if parsed.scheme == "gs":
            return GCSArtifactStore(parsed.netloc, parsed.path.strip("/"))
        if parsed.scheme in ("", "file"):
            raw = ((parsed.netloc + parsed.path) if parsed.netloc else parsed.path) if parsed.scheme else uri
            return LocalArtifactStore(Path(raw).expanduser().resolve())
        raise ValueError(f"unsupported artifact URI {uri!r}; use file:// or gs://")

    @abstractmethod
    def put_file(self, source: Path, key: str, *, overwrite: bool = False) -> None: ...

    @abstractmethod
    def put_bytes(self, payload: bytes, key: str, *, overwrite: bool = False) -> None: ...

    @abstractmethod
    def get_file(self, key: str, destination: Path) -> None: ...

    @abstractmethod
    def get_bytes(self, key: str) -> bytes: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...


class LocalArtifactStore(ArtifactStore):
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if self.root not in path.parents and path != self.root:
            raise ValueError(f"artifact key escapes root: {key}")
        return path

    def put_file(self, source: Path, key: str, *, overwrite: bool = False) -> None:
        target = self._path(key)
        if target.exists() and not overwrite:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + f".tmp.{os.getpid()}")
        shutil.copyfile(source, tmp)
        os.replace(tmp, target)

    def put_bytes(self, payload: bytes, key: str, *, overwrite: bool = False) -> None:
        target = self._path(key)
        if target.exists() and not overwrite:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + f".tmp.{os.getpid()}")
        tmp.write_bytes(payload)
        os.replace(tmp, target)

    def get_file(self, key: str, destination: Path) -> None:
        source = self._path(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp = destination.with_name(destination.name + f".tmp.{os.getpid()}")
        shutil.copyfile(source, tmp)
        os.replace(tmp, destination)

    def get_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).exists()


class GCSArtifactStore(ArtifactStore):
    def __init__(self, bucket: str, prefix: str = ""):
        from google.cloud import storage

        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket)
        self.prefix = prefix.strip("/")

    def _key(self, key: str) -> str:
        return "/".join(x for x in (self.prefix, key.strip("/")) if x)

    def put_file(self, source: Path, key: str, *, overwrite: bool = False) -> None:
        from google.api_core.exceptions import PreconditionFailed

        blob = self.bucket.blob(self._key(key))
        try:
            blob.upload_from_filename(str(source), if_generation_match=None if overwrite else 0,
                                      timeout=3600, checksum="crc32c")
        except PreconditionFailed:
            if overwrite:
                raise

    def put_bytes(self, payload: bytes, key: str, *, overwrite: bool = False) -> None:
        from google.api_core.exceptions import PreconditionFailed

        blob = self.bucket.blob(self._key(key))
        try:
            blob.upload_from_string(payload, if_generation_match=None if overwrite else 0,
                                    timeout=300, checksum="crc32c", content_type="application/json")
        except PreconditionFailed:
            if overwrite:
                raise

    def get_file(self, key: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp = destination.with_name(destination.name + f".tmp.{os.getpid()}")
        self.bucket.blob(self._key(key)).download_to_filename(str(tmp), timeout=3600, checksum="crc32c")
        os.replace(tmp, destination)

    def get_bytes(self, key: str) -> bytes:
        return self.bucket.blob(self._key(key)).download_as_bytes(timeout=300, checksum="crc32c")

    def exists(self, key: str) -> bool:
        return bool(self.bucket.blob(self._key(key)).exists())
