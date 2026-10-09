"""Sequential, persistent BDMO workflow. No implicit territory-ID mapping."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import tempfile
import time

import numpy as np
import pandas as pd

from . import bdmo

VERSION = 'sequential-2'


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    os.replace(temporary, path)


def atomic_parquet(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.parquet')
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def base_panel(path, expected=2016, start='2023-01', end='2024-12'):
    panel = pd.read_parquet(path)
    if panel.index.names != ['territory_id', 'date'] or not panel.index.is_unique:
        raise ValueError('Нужен уникальный индекс territory_id × date из первого ноутбука')
    ids = pd.Index(sorted(panel.index.get_level_values('territory_id').unique()), name='territory_id')
    months = pd.period_range(start, end, freq='M').to_timestamp()
    wanted = pd.MultiIndex.from_product([ids, months], names=panel.index.names)
    if len(ids) != expected or len(months) != 24 or len(panel) != len(wanted) or len(wanted.difference(panel.index)):
        raise ValueError(f'Ожидалось {expected} МО × 24 месяца без пропусков; получено {panel.shape}')
    panel = panel.reindex(wanted)
    if not np.isfinite(panel.to_numpy(dtype=float)).all():
        raise ValueError('В базовой месячной динамике есть пропуски или бесконечности')
    return panel, ids


def load_mapping(path, ids, years):
    """Require a reviewed one-to-one map for every territory and every requested year.

    A static mapping must explicitly declare year; administrative changes must be
    resolved by the author of the crosswalk, never by string/name similarity.
    Blank oktmo is allowed and means unmapped, not a reason to drop the base row.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Нет таблицы territory_id,year,oktmo: {path}. Создайте её по проверенному справочнику.')
    frame = pd.read_csv(path, dtype={'oktmo': 'string'})
    if not {'territory_id', 'year', 'oktmo'} <= set(frame):
        raise ValueError('Таблица соответствия требует territory_id, year, oktmo')
    for key in ['territory_id', 'year']:
        number = pd.to_numeric(frame[key], errors='raise')
        if number.isna().any() or not number.mod(1).eq(0).all():
            raise ValueError(f'Некорректный {key}')
        frame[key] = number.astype('int64')
    nonempty = frame.oktmo.fillna('').str.strip().ne('')
    normalized = frame.oktmo.map(bdmo.normalize_oktmo)
    if (nonempty & normalized.isna()).any():
        raise ValueError('Некорректные ОКТМО: нужны строки из 8 или 11 цифр с ведущими нулями')
    frame['oktmo'] = normalized
    if frame.duplicated(['territory_id', 'year']).any():
        raise ValueError('Повтор territory_id × year в таблице соответствия')
    frame = frame[frame.year.isin(years) & frame.territory_id.isin(ids)]
    if frame.dropna(subset=['oktmo']).duplicated(['oktmo', 'year']).any():
        raise ValueError('Один ОКТМО относится к нескольким базовым МО в одном году')
    grid = pd.MultiIndex.from_product([ids, years], names=['territory_id', 'year']).to_frame(index=False)
    return grid.merge(frame[['territory_id', 'year', 'oktmo']], how='left', validate='one_to_one')


def coverage_against_base(data, mapping, meta, years):
    rows, missing = [], []
    for year in years:
        base = mapping[mapping.year.eq(year)]
        values = data[data.year.eq(year)][['oktmo', 'value']]
        if values.oktmo.duplicated().any():
            raise ValueError('Аудит ожидает один показатель и уникальный ОКТМО × год')
        merged = base.merge(values.dropna(subset=['oktmo']), on='oktmo', how='left', validate='many_to_one')
        observed = int(merged.value.notna().sum())
        rows.append(dict(feature_key=meta['feature_key'], view=meta['view'],
                         requested_indicator=meta['indicator_name'], year=int(year),
                         observed=observed, universe=len(base), coverage=observed / len(base),
                         unmapped=int(base.oktmo.isna().sum()),
                         outside_base=int((~values.oktmo.isin(base.oktmo.dropna())).sum())))
        lost = merged[merged.value.isna()].copy()
        lost['reason'] = np.where(lost.oktmo.isna(), 'no_mapping', 'no_valid_value')
        lost['feature_key'] = meta['feature_key']
        lost['requested_indicator'] = meta['indicator_name']
        missing.append(lost)
    return pd.DataFrame(rows), pd.concat(missing, ignore_index=True)


class IndicatorStore:
    def __init__(self, root, config, session=None):
        self.root = Path(root)
        self.config = config
        self.cache = self.root / config.get('cache_dir', 'data/cache/bdmo')
        self.cache.mkdir(parents=True, exist_ok=True)
        self.session = session or bdmo.make_session()
        self.log_path = self.cache / 'events.jsonl'

    def event(self, status, **fields):
        entry = dict(time=datetime.now(timezone.utc).isoformat(), status=status, **fields)
        with self.log_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(entry, ensure_ascii=False, default=str) + '\n')
        print(status, fields.get('indicator_name', fields.get('url', '')))

    def catalog(self, refresh=False):
        from .recommendations import read_plan, full_catalog
        cache_path = self.cache / 'reference_full.json'
        snapshot = self.root / self.config.get('reference_snapshot', 'data/bdmo_reference.json')
        if refresh:
            links, sheet = bdmo.discover_source(self.config['source_page'], self.session, self.config['request_timeout'])
            reference, info = bdmo.load_indicator_reference(sheet, self.session, self.config['request_timeout'])
            record = dict(info=info, sheet_url=sheet, rows=reference.to_dict('records'))
            atomic_json(cache_path, record)
        elif cache_path.exists():
            record = json.loads(cache_path.read_text())
        elif snapshot.exists():
            record = json.loads(snapshot.read_text())
        else:
            return self.catalog(refresh=True)
        evidence_path = self.root / 'data/indicator_evidence.csv'
        evidence = pd.read_csv(evidence_path, dtype={'indicator_code': str}) if evidence_path.exists() else None
        old = self.config.get('previous_shortlist_codes', [])
        return full_catalog(record['rows'], read_plan(self.root, self.config), evidence, old)

    def import_existing(self):
        from .local_exports import import_export
        reports = []
        for name in self.config.get('local_exports', []):
            path = self.root / name
            if path.exists():
                report = import_export(path, self.cache, self.config['history_years'], self.config['csv_chunk_size'])
                reports.append(report)
                self.event('local_export_indexed', path=str(path))
        return reports

    def load_next(self, queue, mapping=None):
        for meta in queue.to_dict('records'):
            directory = self.cache / 'indicators' / self.key(meta)
            state_path = directory / 'state.json'
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            files = ['raw.parquet', 'canonical.parquet', 'conflicts.parquet']
            complete = state.get('status') == 'complete' and all(
                (directory / f).exists() and digest(directory / f) == state.get('sha256', {}).get(f) for f in files)
            if not complete:
                data, coverage = self.load_one(meta, mapping=mapping)
                return meta, data, coverage
        return None

    def completed_selection(self, queue):
        rows = []
        for meta in queue.to_dict('records'):
            path = self.cache / 'indicators' / self.key(meta) / 'state.json'
            if path.exists() and json.loads(path.read_text()).get('status') == 'complete':
                rows.append(meta)
        return pd.DataFrame(rows, columns=queue.columns)

    def download(self, url, refresh=False):
        directory = self.cache / 'downloads' / hashlib.sha256(url.encode()).hexdigest()
        directory.mkdir(parents=True, exist_ok=True)
        receipt = directory / 'receipt.json'
        if receipt.exists() and not refresh:
            record = json.loads(receipt.read_text())
            path = directory / record['file']
            if path.exists() and digest(path) == record['sha256']:
                self.event('download_cache_hit', url=url)
                return path
        for attempt in range(self.config.get('download_attempts', 3)):
            try:
                with tempfile.TemporaryDirectory(dir=directory) as tmp:
                    path = bdmo._download(url, tmp, self.session, self.config['request_timeout'])
                    final = directory / path.name
                    checksum = digest(path)
                    os.replace(path, final)
                atomic_json(receipt, dict(url=url, file=final.name, sha256=checksum))
                return final
            except Exception as error:
                self.event('download_error', url=url, attempt=attempt + 1, error=str(error))
                if attempt + 1 == self.config.get('download_attempts', 3):
                    raise
                time.sleep(min(2 ** attempt, 8))

    def key(self, meta):
        from .local_exports import local_parts
        fields = ['indicator_code', 'indicator_name', 'feature_key', 'view', 'section', 'download_url',
                  'dimension_filters', 'period_preference']
        content = dict(version='reviewed-3', parser_sha256=digest(bdmo.__file__), workflow_sha256=digest(__file__),
                       meta={k: meta[k] for k in fields if k in meta}, years=self.config['history_years'],
                       annual_periods=self.config.get('annual_periods'), metadata_columns=self.config.get('metadata_columns'),
                       dimension_filters=self.config.get('dimension_filters', {}).get(meta['feature_key'], {}),
                       local_sources=[(str(p), digest(p)) for p in local_parts(self.cache, meta)])
        return hashlib.sha256(json.dumps(content, sort_keys=True, default=str).encode()).hexdigest()[:24]

    def load_one(self, meta, mapping=None, refresh=False):
        meta = dict(meta)
        key = self.key(meta)
        directory = self.cache / 'indicators' / key
        directory.mkdir(parents=True, exist_ok=True)
        state_path = directory / 'state.json'
        files = ['raw.parquet', 'canonical.parquet', 'conflicts.parquet']
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        cached = state.get('status') == 'complete' and all(
            (directory / f).exists() and digest(directory / f) == state.get('sha256', {}).get(f) for f in files)
        if cached and not refresh:
            data = pd.read_parquet(directory / 'canonical.parquet')
            self.event('indicator_cache_hit', **meta)
        else:
            atomic_json(state_path, dict(status='running', meta=meta))
            try:
                from .local_exports import local_parts, code_key
                existing = local_parts(self.cache, meta) if not refresh else []
                archive = None if existing else self.download(meta['download_url'], refresh=refresh)
                parts = []
                file_log = []
                with tempfile.TemporaryDirectory(dir=directory) as tmp:
                    paths = existing or sorted(bdmo._extract(archive, Path(tmp) / 'extracted'))
                    # Archives may contain identical CSV/XLSX copies; read one representation.
                    preferred = {}
                    priority = {'.parquet': 0, '.csv': 1, '.xlsx': 2, '.xls': 3}
                    for candidate in paths:
                        if candidate.suffix.lower() in priority:
                            stem = str(candidate.with_suffix(''))
                            if stem not in preferred or priority[candidate.suffix.lower()] < priority[preferred[stem].suffix.lower()]:
                                preferred[stem] = candidate
                    for path in sorted(preferred.values()):
                        if not bdmo._should_read(path, self.config['history_years']):
                            continue
                        self.event('file_read_started', indicator_name=meta['indicator_name'], file=path.name)
                        frame = bdmo._read_table(path, self.config['history_years'], self.config['csv_chunk_size'])
                        # Filters must be explicit; unknown dimensions remain in detail_score.
                        filters = meta.get('dimension_filters') if isinstance(meta.get('dimension_filters'), dict) else self.config.get('dimension_filters', {}).get(meta['feature_key'], {})
                        if 'indicator_code' in frame and meta.get('indicator_code'):
                            frame = frame[frame.indicator_code.map(code_key).eq(code_key(meta['indicator_code']))]
                        for column, value in filters.items():
                            if column not in frame:
                                raise ValueError(f'Нет колонки фильтра {column}')
                            frame = frame[frame[column].map(bdmo.text_key).eq(bdmo.text_key(value))]
                        clean = bdmo._harmonize(frame, meta['indicator_name'], meta['feature_key'], meta['view'],
                                                meta['section'], self.config['history_years'], path,
                                                metadata_columns=self.config.get('metadata_columns', []) + ['indicator_code'] + list(filters))
                        self.event('file_read_complete', indicator_name=meta['indicator_name'], file=path.name, source_rows=len(frame), matched_rows=len(clean), missing_values=int(clean['value'].isna().sum()) if 'value' in clean else 0)
                        parts.append(clean)
                        file_log.append(dict(file=path.name, source_rows=len(frame), matched_rows=len(clean)))
                if not parts or not any(len(part) for part in parts):
                    raise ValueError('Нет строк выбранного показателя: проверьте название, схему и годы')
                raw = pd.concat(parts, ignore_index=True)
                periods = meta.get('period_preference')
                if not isinstance(periods, list):
                    periods = self.config.get('annual_periods')
                canonical_input = raw
                if isinstance(meta.get('period_preference'), list):
                    ranks = {bdmo.text_key(v): i for i, v in enumerate(periods)}
                    canonical_input = raw.assign(period_rank=raw.period.map(bdmo.text_key).map(ranks))
                    eligible = canonical_input.period_rank.notna() & canonical_input.detail_score.eq(0)
                    canonical_input = canonical_input[eligible].copy()
                    keys = ['oktmo', 'year', 'feature_key', 'view', 'requested_indicator']
                    best = canonical_input.groupby(keys).period_rank.transform('min')
                    canonical_input = canonical_input[canonical_input.period_rank.eq(best)]
                data, conflicts = bdmo.canonicalize(canonical_input, annual_periods=periods)
                data['indicator_code'] = str(meta.get('indicator_code', ''))
                for name, frame in zip(files, [raw, data, conflicts]):
                    atomic_parquet(frame, directory / name)
                atomic_json(directory / 'files.json', file_log)
                atomic_json(state_path, dict(status='complete', meta=meta, rows=len(data),
                                            sha256={f: digest(directory / f) for f in files}))
                self.event('complete', indicator_name=meta['indicator_name'], rows=len(data), rejected=len(conflicts), source='local_export' if existing else 'download')
            except Exception as error:
                atomic_json(state_path, dict(status='failed', meta=meta, error=str(error)))
                self.event('indicator_error', indicator_name=meta['indicator_name'], error=str(error))
                raise
        if mapping is not None:
            coverage, missing = coverage_against_base(data, mapping, meta, self.config['focus_years'])
            atomic_parquet(coverage, directory / 'coverage.parquet')
            atomic_parquet(missing, directory / 'missing.parquet')
        else:
            coverage = pd.DataFrame()
            self.event('coverage_blocked_no_mapping', indicator_name=meta['indicator_name'])
        return data, coverage

    def export(self, candidates, output):
        """Only selected, completed cache entries; never silently export partial runs."""
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        parts, conflicts = [], []
        for meta in candidates.to_dict('records'):
            directory = self.cache / 'indicators' / self.key(meta)
            state_path = directory / 'state.json'
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            if state.get('status') != 'complete':
                raise ValueError(f"Показатель ещё не загружен: {meta['indicator_name']}")
            for name in ['raw.parquet', 'canonical.parquet', 'conflicts.parquet']:
                if not (directory / name).exists() or digest(directory / name) != state['sha256'][name]:
                    raise ValueError('Повреждён кэш; повторите load_one')
            parts.append(pd.read_parquet(directory / 'canonical.parquet'))
            conflicts.append(pd.read_parquet(directory / 'conflicts.parquet'))
        if not parts:
            raise ValueError('Сначала выберите и загрузите хотя бы один показатель')
        data = pd.concat(parts, ignore_index=True).drop_duplicates()
        atomic_parquet(data, output / 'bdmo_long.parquet')
        atomic_parquet(pd.concat(conflicts, ignore_index=True), output / 'conflicts.parquet')
        candidates.to_csv(output / 'indicator_candidates.csv', index=False)
        return data
