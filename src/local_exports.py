"""Reuse existing raw BDMO exports; never infer lost period/dimension metadata."""
from pathlib import Path
import hashlib
import json
import os
import re
import zipfile
from contextlib import contextmanager

import pandas as pd
import pyarrow.parquet as pq

REQUIRED = {'oktmo', 'year', 'indicator_name', 'indicator_value', 'indicator_unit', 'indicator_period'}


def code_key(value):
    text = str(value).strip().removesuffix('.0')
    return text[2:] if text.startswith('Y4') else text


def table_chunks(path, chunk_size=100000):
    path = Path(path)
    if path.suffix.lower() == '.csv':
        yield from pd.read_csv(path, sep=';', dtype=str, chunksize=chunk_size)
    elif path.suffix.lower() == '.parquet':
        for batch in pq.ParquetFile(path).iter_batches(batch_size=chunk_size):
            yield batch.to_pandas()
    elif path.suffix.lower() in {'.xls', '.xlsx'}:
        yield pd.read_excel(path, dtype=str)


def source_identity(path):
    # Content fingerprint prevents reuse after modification even when size/mtime match.
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def import_export(path, cache_dir, years=(2023, 2024), chunk_size=100000):
    """Split a raw file, folder or ZIP once into per-indicator/year-filtered partitions.

    Atomic manifest makes interrupted imports retryable. Original files are read only.
    Legacy canonical exports are reported as unverifiable and are not used as raw data.
    """
    import tempfile
    from .bdmo import _extract
    from .bdmo_workflow import atomic_json
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_dir():
        reports = []
        for f in sorted(path.rglob('*')):
            if f.is_file() and f.suffix.lower() in {'.zip', '.csv', '.parquet', '.xlsx', '.xls'}:
                reports.append(import_export(f, cache_dir, years, chunk_size))
        return reports
    fingerprint = source_identity(path)
    identity = hashlib.sha256(json.dumps([str(path), fingerprint, sorted(years), 'raw-import-v1']).encode()).hexdigest()[:24]
    directory = Path(cache_dir) / 'local_imports' / identity
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / 'manifest.json'
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if saved.get('status') == 'complete' and all((directory / part['file']).exists() and source_identity(directory / part['file']) == part['sha256'] for part in saved['parts']):
            return saved
    parts, skipped = [], []
    with tempfile.TemporaryDirectory(dir=directory) as tmp:
        files = _extract(path, Path(tmp) / 'unpacked') if path.suffix.lower() == '.zip' else [path]
        preferred = {}
        rank = {'.parquet': 0, '.csv': 1, '.xlsx': 2, '.xls': 3}
        for f in files:
            ext = f.suffix.lower()
            if ext in rank:
                key = str(f.with_suffix(''))
                if key not in preferred or rank[ext] < rank[preferred[key].suffix.lower()]:
                    preferred[key] = f
        for f in sorted(preferred.values()):
            for chunk in table_chunks(f, chunk_size):
                if not REQUIRED <= set(chunk):
                    skipped.append({'file': f.name, 'reason': 'missing_raw_columns', 'missing': sorted(REQUIRED-set(chunk))})
                    break
                chunk = chunk[pd.to_numeric(chunk['year'], errors='coerce').isin(years)].copy()
                if chunk.empty:
                    continue
                if 'indicator_code' in chunk:
                    chunk['_source_code'] = chunk.indicator_code.map(code_key)
                else:
                    chunk['_source_code'] = ''
                for (code, name), group in chunk.groupby(['_source_code', 'indicator_name'], dropna=False):
                    filename = f'part_{len(parts):06d}.parquet'
                    temporary = directory / (filename + '.tmp')
                    group.drop(columns='_source_code').to_parquet(temporary, index=False)
                    os.replace(temporary, directory / filename)
                    parts.append(dict(code=code, indicator_name=str(name).strip(), file=filename,
                                      rows=len(group), sha256=source_identity(directory / filename)))
    result = dict(status='complete', source=str(path), source_sha256=fingerprint, years=list(years),
                  parts=parts, skipped=skipped)
    atomic_json(manifest, result)
    return result


def local_parts(cache_dir, meta):
    """Return only committed imports matching exact source identity."""
    found = []
    code = code_key(meta.get('indicator_code', ''))
    for manifest in sorted((Path(cache_dir) / 'local_imports').glob('*/manifest.json')):
        record = json.loads(manifest.read_text())
        if record.get('status') != 'complete':
            continue
        for part in record['parts']:
            if (part['code'] == code and code) or (not part['code'] and part['indicator_name'] == meta['indicator_name'].strip()):
                path = manifest.parent / part['file']
                if not path.exists() or source_identity(path) != part['sha256']:
                    raise ValueError(f'Повреждён импорт локального экспорта: {path}')
                found.append(path)
    return found
