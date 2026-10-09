from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup


CATEGORY_EN = {
    'Все категории': 'total',
    'Продовольствие': 'food',
    'Здоровье': 'health',
    'Маркетплейсы': 'marketplace',
    'Общественное питание': 'dining',
    'Транспорт': 'transport',
}

PARTS = ['food', 'health', 'marketplace', 'dining', 'transport']
SHARE_COLUMNS = [f'share_{name}' for name in [*PARTS, 'other']]


def _project_path(root, value):
    path = Path(value)
    return path if path.is_absolute() else Path(root) / path


def discover_downloads(source_page, filenames, timeout=60):
    response = requests.get(
        source_page,
        timeout=timeout,
        headers={'User-Agent': 'Mozilla/5.0'},
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')
    found = {}

    for anchor in soup.find_all('a', href=True):
        href = urljoin(source_page, anchor['href'])
        text = f"{anchor.get_text(' ', strip=True)} {href}".casefold()

        for key, filename in filenames.items():
            if filename.casefold() in text:
                found[key] = href

    return found


def ensure_source_files(config, root='.'):
    raw_dir = _project_path(root, config['raw_dir'])
    raw_dir.mkdir(parents=True, exist_ok=True)

    filenames = config['files']
    missing = {
        key: filename
        for key, filename in filenames.items()
        if not (raw_dir / filename).exists()
    }
    if not missing:
        return {key: raw_dir / filename for key, filename in filenames.items()}

    urls = dict(config.get('urls', {}))
    unresolved = [key for key in missing if not urls.get(key)]

    if unresolved:
        discovered = discover_downloads(
            config['source_page'],
            {key: missing[key] for key in unresolved},
        )
        urls.update(discovered)

    still_missing = [key for key in missing if not urls.get(key)]
    if still_missing:
        names = ', '.join(filenames[key] for key in still_missing)
        raise RuntimeError(
            'Не удалось автоматически найти ссылки на файлы: '
            f'{names}. Положи исходные файлы в {raw_dir} '
            'или укажи прямые URL в configs/sber.json.'
        )

    for key, filename in missing.items():
        response = requests.get(
            urls[key],
            timeout=120,
            headers={'User-Agent': 'Mozilla/5.0'},
        )
        response.raise_for_status()
        (raw_dir / filename).write_bytes(response.content)

    return {key: raw_dir / filename for key, filename in filenames.items()}


def load_sber(config, root='.'):
    paths = ensure_source_files(config, root)

    consumption = pd.read_parquet(paths['consumption'])
    market_access = pd.read_parquet(paths['market_access'])
    connection = pd.read_parquet(paths['connection'])

    return consumption, market_access, connection


def _integer_id(series, name):
    values = pd.to_numeric(series, errors='raise')
    if values.isna().any():
        raise ValueError(f'В {name} есть пропуски.')
    if not np.equal(values, np.floor(values)).all():
        raise ValueError(f'В {name} есть нецелые значения.')
    return values.astype('int64')


def prepare_consumption(consumption, calendar):
    data = consumption.copy()
    if 'value' in data.columns and 'consumption' not in data.columns:
        data = data.rename(columns={'value': 'consumption'})

    required = {'territory_id', 'date', 'category', 'consumption'}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f'В consumption нет колонок: {sorted(missing)}')

    data['territory_id'] = _integer_id(data['territory_id'], 'territory_id')
    data['date'] = pd.to_datetime(data['date'], errors='raise').dt.to_period('M').dt.to_timestamp()

    if data[['territory_id', 'date', 'category']].isna().any().any():
        raise ValueError('В ключах consumption есть пропуски.')
    if data.duplicated(['territory_id', 'date', 'category']).any():
        raise ValueError('В consumption есть дубли territory_id × date × category.')

    data['consumption'] = pd.to_numeric(data['consumption'], errors='coerce')
    if not np.isfinite(data['consumption']).all():
        raise ValueError('В consumption есть нечисловые или бесконечные значения.')

    data['category'] = data['category'].map(CATEGORY_EN)
    if data['category'].isna().any():
        unknown = sorted(consumption.loc[data['category'].isna(), 'category'].astype(str).unique())
        raise ValueError(f'Неизвестные категории расходов: {unknown}')

    panel_all = data.pivot(
        index=['territory_id', 'date'],
        columns='category',
        values='consumption',
    )
    panel_all = panel_all[['total', *PARTS]].sort_index()
    panel_all.columns.name = None

    if panel_all.isna().any().any():
        raise ValueError('После pivot в расходах появились пропуски.')
    if not panel_all['total'].gt(0).all():
        raise ValueError('Общие расходы должны быть положительными.')
    if not panel_all[PARTS].ge(0).all().all():
        raise ValueError('Расходы по категориям не могут быть отрицательными.')

    panel_all['other'] = panel_all['total'] - panel_all[PARTS].sum(axis=1)
    if panel_all['other'].lt(-1e-9).any():
        raise ValueError('Сумма пяти категорий местами больше общих расходов.')
    panel_all['other'] = panel_all['other'].clip(lower=0)

    for part in [*PARTS, 'other']:
        panel_all[f'share_{part}'] = panel_all[part] / panel_all['total']

    months = pd.date_range(
        calendar['start'],
        calendar['end'],
        freq=calendar.get('freq', 'MS'),
    )
    observed = pd.DatetimeIndex(
        sorted(panel_all.index.get_level_values('date').unique())
    )
    if not observed.equals(months):
        raise ValueError(
            'Календарь расходов не совпадает с конфигурацией: '
            f'ожидалось {months.min():%Y-%m}..{months.max():%Y-%m}, '
            f'получено {observed.min():%Y-%m}..{observed.max():%Y-%m}.'
        )

    month_counts = panel_all.groupby(level='territory_id').size()
    complete_ids = month_counts.index[month_counts.eq(len(months))]
    panel = panel_all[
        panel_all.index.get_level_values('territory_id').isin(complete_ids)
    ].copy()

    return panel_all, panel, complete_ids, months


def prepare_market_access(market_access, territory_ids):
    data = market_access.copy()
    required = {'territory_id', 'market_access'}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f'В market_access нет колонок: {sorted(missing)}')

    data['territory_id'] = _integer_id(data['territory_id'], 'territory_id')
    data['market_access'] = pd.to_numeric(data['market_access'], errors='coerce')

    if data['territory_id'].duplicated().any():
        raise ValueError('В market_access есть дубли territory_id.')
    if not np.isfinite(data['market_access'].dropna()).all():
        raise ValueError('В market_access есть бесконечные значения.')

    matrix = data.set_index('territory_id')[['market_access']]
    return matrix.reindex(pd.Index(territory_ids, name='territory_id')).sort_index()


def prepare_transport_edges(connection, territory_ids):
    data = connection.copy()
    required = {'territory_id_x', 'territory_id_y', 'distance', 'type'}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f'В connection нет колонок: {sorted(missing)}')

    for column in ['territory_id_x', 'territory_id_y']:
        data[column] = _integer_id(data[column], column)

    data['distance'] = pd.to_numeric(data['distance'], errors='coerce')

    if data[['territory_id_x', 'territory_id_y', 'type']].isna().any().any():
        raise ValueError('В ключах connection есть пропуски.')
    if not data['type'].isin(['highway', 'railway']).all():
        raise ValueError('В connection найден неизвестный тип связи.')
    if not np.isfinite(data['distance']).all() or data['distance'].lt(0).any():
        raise ValueError('Расстояния должны быть конечными и неотрицательными.')
    if data['territory_id_x'].eq(data['territory_id_y']).any():
        raise ValueError('В connection есть петли territory_id_x = territory_id_y.')

    territory_ids = pd.Index(territory_ids)
    selected = data[
        data['territory_id_x'].isin(territory_ids)
        & data['territory_id_y'].isin(territory_ids)
    ].copy()

    layers = []
    for kind, layer in selected.groupby('type', observed=True):
        x = layer['territory_id_x'].to_numpy()
        y = layer['territory_id_y'].to_numpy()
        pairs = pd.DataFrame({
            'u': np.minimum(x, y),
            'v': np.maximum(x, y),
            'distance': layer['distance'].to_numpy(),
        })

        spread = pairs.groupby(['u', 'v'])['distance'].agg(['min', 'max'])
        if not np.allclose(spread['min'], spread['max']):
            raise ValueError(f'Для {kind} найдены разные расстояния у одной пары МО.')

        pairs = pairs.drop_duplicates(['u', 'v'])
        pairs['type'] = kind
        layers.append(pairs)

    if not layers:
        return pd.DataFrame(columns=['u', 'v', 'type', 'distance', 'zero_distance'])

    edges = pd.concat(layers, ignore_index=True)
    edges['zero_distance'] = edges['distance'].eq(0)
    edges = edges[['u', 'v', 'type', 'distance', 'zero_distance']]
    edges['type'] = edges['type'].astype('category')

    if not edges['u'].lt(edges['v']).all():
        raise ValueError('После нормализации в edges есть некорректные пары.')
    if edges.duplicated(['u', 'v', 'type']).any():
        raise ValueError('После нормализации в edges остались дубли.')

    return edges.sort_values(['type', 'u', 'v']).reset_index(drop=True)


def consumer_matrix(panel):
    matrix = panel[SHARE_COLUMNS].copy()
    if matrix.isna().any().any() or not np.isfinite(matrix).all().all():
        raise ValueError('В X_consumer есть пропуски или бесконечные значения.')
    return matrix


def spending_summary(panel_all, panel, market_access):
    ids = panel.index.get_level_values('territory_id').unique()
    result = pd.DataFrame(index=pd.Index(ids, name='territory_id'))

    monthly_rank = panel_all['total'].groupby(level='date').rank(pct=True) * 100
    result['spending_percentile'] = monthly_rank.groupby(level='territory_id').mean()

    log_change = np.log(panel['total']).unstack('date').diff(axis=1).iloc[:, 1:]
    centered = log_change.sub(log_change.median(axis=0), axis=1)
    result['monthly_volatility'] = 1.4826 * centered.abs().median(axis=1)

    result = result.join(market_access, how='left')
    return result.sort_index()


def build_sber_views(consumption, market_access, connection, calendar):
    panel_all, panel, complete_ids, months = prepare_consumption(consumption, calendar)
    x_consumer = consumer_matrix(panel)
    x_market_access = prepare_market_access(market_access, complete_ids)
    edges = prepare_transport_edges(connection, complete_ids)
    x_context = spending_summary(panel_all, panel, x_market_access)

    diagnostics = {
        'months': len(months),
        'municipalities_with_spending': panel_all.index.get_level_values('territory_id').nunique(),
        'municipalities_complete': len(complete_ids),
        'consumer_rows': len(x_consumer),
        'consumer_features': x_consumer.shape[1],
        'market_access_coverage': float(x_market_access['market_access'].notna().mean()),
        'transport_edges': len(edges),
    }

    return {
        'X_consumer': x_consumer,
        'X_market_access': x_market_access,
        'X_sber_context': x_context,
        'transport_edges': edges,
        'diagnostics': diagnostics,
    }


def save_sber_views(views, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = {
        'X_consumer': output_dir / 'X_consumer.parquet',
        'X_market_access': output_dir / 'X_market_access.parquet',
        'X_sber_context': output_dir / 'X_sber_context.parquet',
        'transport_edges': output_dir / 'transport_edges.parquet',
    }

    for name, path in files.items():
        views[name].to_parquet(path)

    return files
