"""Non-overwriting stage entry and explicit new-source provenance."""
from pathlib import Path
import json

from experiments.seoyeon import comparison_io as io


def assert_new_outputs(directory, names):
    existing = [name for name in names if (Path(directory) / name).exists()]
    if existing:
        raise FileExistsError(f'Existing stage output; use a new run: {existing}')


def source_hashes(names):
    return {name: io.sha256(Path(__file__).parent / name) for name in names}


def verify_sources(before):
    after = source_hashes(before)
    if before != after:
        raise ValueError('Comparison implementation changed during stage execution')
    return dict(source_before=before, source_after=after, source_preserved=True)


def verified_outputs(directory, manifest_name, names):
    """Verify completed artifacts against the producer's own output hashes."""
    directory = Path(directory)
    record = json.loads((directory / manifest_name).read_text())
    if record.get('status') != 'passed':
        raise ValueError('Producer verification is incomplete')
    expected = record.get('outputs_sha256', {})
    if any(expected.get(name) != io.sha256(directory / name) for name in names):
        raise ValueError('Verified producer output checksum mismatch')
    return record
