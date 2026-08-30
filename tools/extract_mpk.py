#!/usr/bin/env python3
"""Safely extract MAGES MPK archives and write a reproducible CSV inventory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


HEADER_SIZE = 0x40
ENTRY_SIZE = 0x100
NAME_OFFSET = 0x20
COPY_CHUNK_SIZE = 1024 * 1024
OGG_TAIL_SIZE = 128 * 1024


@dataclass(frozen=True)
class Entry:
    table_index: int
    archive_index: int
    flags: int
    offset: int
    stored_size: int
    original_size: int
    filename: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(COPY_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_entries(archive: Path) -> list[Entry]:
    archive_size = archive.stat().st_size
    with archive.open("rb") as handle:
        header = handle.read(HEADER_SIZE)
        if len(header) != HEADER_SIZE or header[:4] != b"MPK\0":
            raise ValueError(f"Not a supported MPK archive: {archive}")

        entry_count = struct.unpack_from("<I", header, 0x08)[0]
        table_end = HEADER_SIZE + entry_count * ENTRY_SIZE
        if table_end > archive_size:
            raise ValueError("MPK entry table extends beyond the archive")

        entries: list[Entry] = []
        seen_names: set[str] = set()
        for table_index in range(entry_count):
            raw = handle.read(ENTRY_SIZE)
            if len(raw) != ENTRY_SIZE:
                raise ValueError(f"Truncated entry table at index {table_index}")

            flags, archive_index, offset, stored_size, original_size = struct.unpack_from(
                "<IIQQQ", raw, 0
            )
            name_raw = raw[NAME_OFFSET:].split(b"\0", 1)[0]
            try:
                filename = name_raw.decode("ascii")
            except UnicodeDecodeError:
                filename = name_raw.decode("shift_jis")

            relative = safe_relative_path(filename)
            folded = relative.as_posix().casefold()
            if folded in seen_names:
                raise ValueError(f"Duplicate output filename: {filename!r}")
            seen_names.add(folded)

            if offset < table_end or offset + stored_size > archive_size:
                raise ValueError(
                    f"Invalid payload range for entry {table_index}: "
                    f"offset={offset}, size={stored_size}"
                )
            if stored_size != original_size:
                raise ValueError(
                    f"Compressed MPK entry {filename!r} is not supported "
                    f"({stored_size} != {original_size})"
                )

            entries.append(
                Entry(
                    table_index=table_index,
                    archive_index=archive_index,
                    flags=flags,
                    offset=offset,
                    stored_size=stored_size,
                    original_size=original_size,
                    filename=filename,
                )
            )
    return entries


def safe_relative_path(filename: str) -> PurePosixPath:
    normalized = filename.replace("\\", "/")
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"Unsafe archive path: {filename!r}")
    return path


def ogg_metadata(first_bytes: bytes, tail_bytes: bytes) -> tuple[str, str, str]:
    sample_rate = ""
    channels = ""
    duration_ms = ""

    marker = first_bytes.find(b"\x01vorbis")
    if marker >= 0 and marker + 16 <= len(first_bytes):
        channels_value = first_bytes[marker + 11]
        sample_rate_value = struct.unpack_from("<I", first_bytes, marker + 12)[0]
        if channels_value and sample_rate_value:
            channels = str(channels_value)
            sample_rate = str(sample_rate_value)

            search_end = len(tail_bytes)
            while True:
                page = tail_bytes.rfind(b"OggS", 0, search_end)
                if page < 0:
                    break
                if page + 27 <= len(tail_bytes) and tail_bytes[page + 4] == 0:
                    segment_count = tail_bytes[page + 26]
                    header_end = page + 27 + segment_count
                    if header_end <= len(tail_bytes):
                        body_size = sum(tail_bytes[page + 27 : header_end])
                        if header_end + body_size <= len(tail_bytes):
                            granule = struct.unpack_from("<Q", tail_bytes, page + 6)[0]
                            if granule != 0xFFFFFFFFFFFFFFFF:
                                duration_ms = str(round(granule * 1000 / sample_rate_value))
                                break
                search_end = page

    return sample_rate, channels, duration_ms


def extract_entry(archive_handle, entry: Entry, destination: Path) -> tuple[str, bytes, bytes]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    digest = hashlib.sha256()
    first = bytearray()
    tail = bytearray()
    remaining = entry.stored_size
    archive_handle.seek(entry.offset)

    with partial.open("wb") as output:
        while remaining:
            chunk = archive_handle.read(min(COPY_CHUNK_SIZE, remaining))
            if not chunk:
                raise IOError(f"Unexpected end of archive while reading {entry.filename}")
            output.write(chunk)
            digest.update(chunk)
            if len(first) < 256:
                first.extend(chunk[: 256 - len(first)])
            tail.extend(chunk)
            if len(tail) > OGG_TAIL_SIZE:
                del tail[:-OGG_TAIL_SIZE]
            remaining -= len(chunk)

    if destination.exists():
        destination.unlink()
    partial.replace(destination)
    return digest.hexdigest(), bytes(first), bytes(tail)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing extracted files; otherwise matching-size files are reused",
    )
    args = parser.parse_args()

    archive = args.archive.resolve()
    output = args.output.resolve()
    manifest = args.manifest.resolve()
    entries = read_entries(archive)
    archive_digest = sha256_file(archive)
    print(f"Archive: {archive}", flush=True)
    print(f"SHA-256: {archive_digest}", flush=True)
    print(f"Entries: {len(entries)}", flush=True)

    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest_partial = manifest.with_name(manifest.name + ".part")
    fieldnames = [
        "archive_path",
        "archive_sha256",
        "table_index",
        "archive_index",
        "flags",
        "filename",
        "offset",
        "stored_size",
        "original_size",
        "extracted_path",
        "sha256",
        "codec",
        "sample_rate",
        "channels",
        "duration_ms",
        "status",
    ]

    with archive.open("rb") as archive_handle, manifest_partial.open(
        "w", encoding="utf-8", newline=""
    ) as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=fieldnames)
        writer.writeheader()
        for number, entry in enumerate(entries, start=1):
            relative = safe_relative_path(entry.filename)
            destination = output.joinpath(*relative.parts)
            status = "extracted"

            if destination.exists() and not args.overwrite:
                if destination.stat().st_size != entry.original_size:
                    raise FileExistsError(
                        f"Existing file has the wrong size (use --overwrite): {destination}"
                    )
                digest = sha256_file(destination)
                with destination.open("rb") as existing:
                    first = existing.read(256)
                    existing.seek(max(0, entry.original_size - OGG_TAIL_SIZE))
                    tail = existing.read()
                status = "reused"
            else:
                digest, first, tail = extract_entry(archive_handle, entry, destination)

            codec = "ogg_vorbis" if first.startswith(b"OggS") and b"vorbis" in first else "unknown"
            sample_rate, channels, duration_ms = ogg_metadata(first, tail)
            writer.writerow(
                {
                    "archive_path": str(archive),
                    "archive_sha256": archive_digest,
                    "table_index": entry.table_index,
                    "archive_index": entry.archive_index,
                    "flags": entry.flags,
                    "filename": entry.filename,
                    "offset": entry.offset,
                    "stored_size": entry.stored_size,
                    "original_size": entry.original_size,
                    "extracted_path": str(destination),
                    "sha256": digest,
                    "codec": codec,
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "duration_ms": duration_ms,
                    "status": status,
                }
            )
            if number % 500 == 0 or number == len(entries):
                print(f"Processed {number}/{len(entries)}", flush=True)

        csv_handle.flush()
        os.fsync(csv_handle.fileno())

    manifest_partial.replace(manifest)
    print(f"Manifest: {manifest}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
