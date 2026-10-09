import numpy as np
import pandas as pd


def indicator_coverage(data, candidates, focus_years, base_mapping=None):
    from .bdmo_workflow import coverage_against_base
    if base_mapping is None:
        raise ValueError('Нужна проверенная таблица соответствия базовым МО; знаменатель нельзя брать из БДМО')
    rows = []
    for meta in candidates.to_dict('records'):
        local = data[data.feature_key.eq(meta['feature_key']) & data.view.eq(meta['view']) &
                     data.requested_indicator.eq(meta['indicator_name'])]
        coverage, _ = coverage_against_base(local, base_mapping, meta, focus_years)
        units = sorted(set(local.loc[local.value.notna(), 'unit'].fillna('')))
        row = dict(feature_key=meta['feature_key'], view=meta['view'],
                   requested_indicator=meta['indicator_name'], match_score=meta.get('match_score', 0),
                   units=units, n_units=len(units), unit_missing=not units or '' in units)
        row.update({int(r.year): r.coverage for r in coverage.itertuples()})
        row['min_focus_coverage'] = min(row[y] for y in focus_years)
        rows.append(row)
    return pd.DataFrame(rows)


def assign_status(coverage, core_min=0.90, sensitivity_min=0.60):
    result = coverage.copy()
    result['status'] = np.select(
        [
            result['min_focus_coverage'].ge(core_min),
            result['min_focus_coverage'].ge(sensitivity_min),
        ],
        ['core', 'sensitivity'],
        default='insufficient_coverage',
    )
    result.loc[result['n_units'].ne(1) | result.get('unit_missing', False), 'status'] = 'check_unit'
    return result


def choose_indicators(audit, overrides=None):
    overrides = overrides or {}
    rows = []

    for feature_key, group in audit.groupby('feature_key', sort=False):
        group = group.copy()
        override = overrides.get(feature_key)

        if override:
            chosen = group[group['requested_indicator'].eq(override)]
            if chosen.empty:
                raise ValueError(f'Для {feature_key} не найден indicator override: {override}')
            rows.append(chosen.iloc[0])
            continue

        core = group[group['status'].eq('core')]
        sensitivity = group[group['status'].eq('sensitivity')]

        if len(core):
            pool = core
        elif len(sensitivity):
            pool = sensitivity
        else:
            pool = group

        pool = pool.sort_values(
            ['match_score', 'min_focus_coverage'],
            ascending=[False, False],
        )
        rows.append(pool.iloc[0])

    if not rows:
        return audit.iloc[:0].copy()
    selected = pd.DataFrame(rows).reset_index(drop=True)
    return selected.sort_values(['view', 'feature_key']).reset_index(drop=True)


def basic_checks(data, focus_years):
    keys = ['oktmo', 'year', 'feature_key', 'requested_indicator']
    return pd.DataFrame([
        {'check': 'rows', 'value': len(data)},
        {'check': 'municipalities', 'value': data['oktmo'].nunique()},
        {'check': 'years', 'value': ', '.join(map(str, sorted(data['year'].unique())))},
        {'check': 'focus_years_present', 'value': all(year in set(data['year']) for year in focus_years)},
        {'check': 'missing_oktmo', 'value': int(data['oktmo'].isna().sum())},
        {'check': 'duplicate_keys', 'value': int(data.duplicated(keys).sum())},
        {'check': 'missing_values', 'value': int(data['value'].isna().sum())},
    ])
