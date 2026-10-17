#!/usr/bin/env python3
"""Provision and verify the pinned offline GPT-OSS Harmony vocabulary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from urllib.request import urlopen

# OpenAI Harmony's O200kHarmony encoding uses this O200kBase vocabulary.
# Source: openai/harmony, src/tiktoken_ext/public_encodings.rs.
VOCAB_FILENAME = "o200k_base.tiktoken"
VOCAB_URL = (
    "https://openaipublic.blob.core.windows.net/encodings/" + VOCAB_FILENAME
)
VOCAB_SHA256 = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
PROTOCOL = "offline-harmony-vocab-1"
MAX_DOWNLOAD_BYTES = 16 * 1024 * 1024


def _verify_file(path: Path) -> None:
    """Reject absent or altered vocabulary bytes before loading any tokenizer."""
    if not path.is_file():
        raise ValueError(f"Missing bundled Harmony vocabulary: {path}")
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != VOCAB_SHA256:
        raise ValueError(f"Harmony vocabulary SHA-256 mismatch: {path}")


def verify(directory: Path) -> dict[str, str]:
    """Verify the one named asset needed for GPT-OSS without network access."""
    _verify_file(directory / VOCAB_FILENAME)
    return {
        "protocol": PROTOCOL,
        "filename": VOCAB_FILENAME,
        "sha256": VOCAB_SHA256,
    }


def download(directory: Path) -> dict[str, str]:
    """Explicitly fetch the public asset and publish it only after verification."""
    destination = directory / VOCAB_FILENAME
    if destination.exists():
        return verify(directory)
    directory.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
            temporary = Path(stream.name)
            with urlopen(VOCAB_URL, timeout=60) as response:
                count = 0
                while chunk := response.read(1024 * 1024):
                    count += len(chunk)
                    if count > MAX_DOWNLOAD_BYTES:
                        raise ValueError(
                            "Harmony vocabulary download exceeds size limit"
                        )
                    stream.write(chunk)
        _verify_file(temporary)
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return verify(directory)


def probe(directory: Path) -> dict[str, str | int | bool]:
    """Load and render Harmony on CPU using the explicit, verified local asset."""
    receipt: dict[str, str | int | bool] = dict(verify(directory))
    configured = os.environ.get("TIKTOKEN_ENCODINGS_BASE")
    if not configured or Path(configured).resolve() != directory.resolve():
        raise ValueError("Set TIKTOKEN_ENCODINGS_BASE to the verified asset directory")
    try:
        harmony = import_module("openai_harmony")
        encoding = harmony.load_harmony_encoding(
            harmony.HarmonyEncodingName.HARMONY_GPT_OSS
        )
        conversation = harmony.Conversation.from_messages(
            [harmony.Message.from_role_and_content(harmony.Role.USER, "Hello.")]
        )
        tokens = encoding.render_conversation_for_completion(
            conversation, harmony.Role.ASSISTANT
        )
        if not tokens or any(type(token) is not int for token in tokens):
            raise ValueError("Harmony rendering did not produce integer tokens")
        package_version = version("openai-harmony")
    except Exception as error:
        raise RuntimeError(
            f"Offline Harmony CPU render preflight failed: {error}"
        ) from error
    receipt.update(
        openai_harmony=package_version,
        rendered_tokens=len(tokens),
        harmony_render_passed=True,
        generation_requests=0,
    )
    return receipt


def main() -> None:
    """Keep provisioning separate from the offline verification and probe paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("download", "verify", "probe"))
    parser.add_argument("--directory", required=True, type=Path)
    args = parser.parse_args()
    action = {"download": download, "verify": verify, "probe": probe}[args.command]
    try:
        receipt = action(args.directory.resolve())
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"{error}\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
