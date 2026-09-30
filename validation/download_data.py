from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path

import requests

from validation.paths import DATASET_ROOT


ZENODO_RECORD_ID = 23041223
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

MANIFEST_NAME = "SHA256SUMS.txt"
README_NAME = "README.md"

ARCHIVE_NAMES = [
    f"freemocap_validation_dataset_v1.0_sub-{i:03d}.zip"
    for i in range(1, 7)
]

CHUNK_SIZE = 1024 * 1024


def get_record_files() -> dict[str, dict]:
    response = requests.get(ZENODO_API_URL, timeout=30)

    if response.status_code == 404:
        raise RuntimeError(
            f"Zenodo record {ZENODO_RECORD_ID} is not publicly available. "
            "Has the record been published?"
        )

    response.raise_for_status()
    record = response.json()

    return {
        file_info["key"]: file_info
        for file_info in record["files"]
    }


def get_download_url(file_info: dict) -> str:
    links = file_info["links"]

    for key in ("download", "content", "self"):
        if key in links:
            return links[key]

    raise RuntimeError(
        f"No download URL found for {file_info['key']}"
    )


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")

    with requests.get(url, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        with temp_path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue

                file.write(chunk)
                downloaded += len(chunk)

                if total_size:
                    percent = downloaded / total_size * 100
                    print(
                        f"\r  {percent:5.1f}% "
                        f"({downloaded / 1e6:.1f} / {total_size / 1e6:.1f} MB)",
                        end="",
                        flush=True,
                    )

    if total_size:
        print()

    temp_path.replace(destination)


def read_sha256_manifest(manifest_path: Path) -> dict[str, str]:
    checksums = {}

    for line in manifest_path.read_text().splitlines():
        line = line.strip()

        if not line:
            continue

        checksum, filename = line.split(maxsplit=1)
        checksums[filename.lstrip("*")] = checksum.lower()

    return checksums


def calculate_sha256(file_path: Path) -> str:
    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(CHUNK_SIZE), b""):
            digest.update(chunk)

    return digest.hexdigest()


def verify_sha256(file_path: Path, expected_checksum: str) -> None:
    actual_checksum = calculate_sha256(file_path)

    if actual_checksum != expected_checksum:
        raise RuntimeError(
            f"SHA-256 mismatch for {file_path.name}\n"
            f"Expected: {expected_checksum}\n"
            f"Actual:   {actual_checksum}"
        )


def extract_archive(archive_path: Path, destination: Path) -> None:
    destination = destination.resolve()

    with zipfile.ZipFile(archive_path, "r") as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()

            if not target.is_relative_to(destination):
                raise RuntimeError(
                    f"Unsafe ZIP path: {member.filename}"
                )

        archive.extractall(destination)


def download_dataset(
    destination: Path = DATASET_ROOT,
    force: bool = False,
) -> None:
    destination = destination.resolve()

    if destination.exists() and any(destination.iterdir()):
        if not force:
            raise FileExistsError(
                f"Dataset directory already exists and is not empty:\n"
                f"{destination}\n\n"
                "Use --force to replace it."
            )

        shutil.rmtree(destination)

    destination.mkdir(parents=True, exist_ok=True)

    data_dir = destination / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    download_dir = destination / ".downloads"
    download_dir.mkdir()

    print(f"Zenodo record: {ZENODO_RECORD_ID}")
    print(f"Destination: {destination}")
    print()

    files = get_record_files()

    required_files = {
        MANIFEST_NAME,
        README_NAME,
        *ARCHIVE_NAMES,
    }

    missing_files = required_files - set(files)

    if missing_files:
        raise RuntimeError(
            "Zenodo record is missing required files:\n"
            + "\n".join(f"  - {name}" for name in sorted(missing_files))
        )

    print(f"Downloading {MANIFEST_NAME}")
    manifest_path = destination / MANIFEST_NAME
    download_file(
        get_download_url(files[MANIFEST_NAME]),
        manifest_path,
    )

    print(f"Downloading {README_NAME}")
    download_file(
        get_download_url(files[README_NAME]),
        destination / README_NAME,
    )

    checksums = read_sha256_manifest(manifest_path)

    for index, archive_name in enumerate(ARCHIVE_NAMES, start=1):
        print()
        print(f"[{index}/{len(ARCHIVE_NAMES)}] {archive_name}")

        if archive_name not in checksums:
            raise RuntimeError(
                f"{archive_name} is missing from {MANIFEST_NAME}"
            )

        archive_path = download_dir / archive_name

        download_file(
            get_download_url(files[archive_name]),
            archive_path,
        )

        print("  Verifying SHA-256...")
        verify_sha256(
            archive_path,
            checksums[archive_name],
        )

        print("  Extracting...")
        extract_archive(
            archive_path,
            data_dir,
        )

        archive_path.unlink()
        print("  Done")

    shutil.rmtree(download_dir, ignore_errors=True)

    print()
    print("Dataset download complete")
    print(f"Data directory: {data_dir}")