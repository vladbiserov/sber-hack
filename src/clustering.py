"""Prepare aligned monthly dynamics and annual context without synthesizing months."""
from pathlib import Path
import numpy as np
import pandas as pd
from .bdmo_workflow import base_panel, atomic_json


def prepare_dynamics(panel, output):
    ids = pd.Index(sorted(panel.index.get_level_values('territory_id').unique()), name='territory_id')
    months = pd.DatetimeIndex(sorted(panel.index.get_level_values('date').unique()))
    if len(months) != 24 or not months.equals(pd.date_range(months.min(), periods=24, freq='MS')):
        raise ValueError('Нужны ровно 24 последовательных месяца')
    index = pd.MultiIndex.from_product([ids, months], names=['territory_id', 'date'])
    if not panel.index.is_unique or len(panel) != len(index) or len(index.difference(panel.index)):
        raise ValueError('Неполная или дублированная месячная панель')
    values = panel.reindex(index).to_numpy(dtype=float).reshape(len(ids), 24, panel.shape[1])
    if not np.isfinite(values).all():
        raise ValueError('Месячная динамика содержит пропуски')
    columns = [f'{feature}__{month:%Y-%m}' for month in months for feature in panel.columns]
    wide = pd.DataFrame(values.reshape(len(ids), -1), index=ids, columns=columns)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(output / 'X_consumer_24m.parquet')
    np.save(output / 'consumer_24m.npy', values, allow_pickle=False)
    atomic_json(output / 'consumer_axes.json', dict(territory_ids=ids.tolist(),
                months=[d.strftime('%Y-%m') for d in months], features=panel.columns.tolist(),
                shape=list(values.shape)))
    return wide


def annual_context(data, selected, mapping, ids, years, accepted=('core',)):
    """Raw levels and two-year differences; never fill missing data or repeat annually into months."""
    matrices, dictionaries = {}, []
    for view, group in selected[selected.status.isin(accepted)].groupby('view'):
        matrix = pd.DataFrame(index=ids)
        for row in group.itertuples():
            local = data[data.feature_key.eq(row.feature_key) & data.view.eq(row.view) &
                         data.requested_indicator.eq(row.requested_indicator)]
            units = set(local.loc[local.value.notna(), 'unit'].fillna(''))
            if len(units) != 1 or '' in units:
                raise ValueError(f'Неоднозначная единица: {row.feature_key}')
            if local.duplicated(['oktmo', 'year']).any():
                raise ValueError('Неуникальный показатель ОКТМО × год')
            joined = mapping.merge(local[['oktmo', 'year', 'value']], on=['oktmo', 'year'],
                                   how='left', validate='many_to_one')
            for year in years:
                column = f'{row.feature_key}__{year}'
                matrix[column] = joined[joined.year.eq(year)].set_index('territory_id').value.reindex(ids)
                dictionaries.append(dict(view=view, column=column, unit=next(iter(units)),
                                         indicator=row.requested_indicator, transformation='annual level'))
            if len(years) == 2:
                column = f'{row.feature_key}__delta_{years[0]}_{years[1]}'
                matrix[column] = matrix[f'{row.feature_key}__{years[1]}'] - matrix[f'{row.feature_key}__{years[0]}']
                dictionaries.append(dict(view=view, column=column, unit=next(iter(units)),
                                         indicator=row.requested_indicator, transformation='difference, only paired observations'))
        matrices[view] = matrix.replace([np.inf, -np.inf], np.nan)
    return matrices, pd.DataFrame(dictionaries)


def preprocess_context(matrix, *, impute=False):
    """Explicit opt-in median imputation and robust scaling; return missing mask and fit parameters.

    All-missing and constant columns are excluded, rows are never dropped. This is
    fit-on-current-cohort preprocessing for unsupervised exploration, not a temporal
    validation pipeline. Fit separately on training data for predictive evaluation.
    """
    matrix = matrix.replace([np.inf, -np.inf], np.nan).astype(float)
    mask = matrix.isna()
    if mask.any().any() and not impute:
        raise ValueError('Есть пропуски. Используйте маску либо явно укажите impute=True после аудита.')
    medians = matrix.median()
    iqr = matrix.quantile(.75) - matrix.quantile(.25)
    usable = medians.notna() & iqr.gt(0)
    if not usable.any():
        raise ValueError('Нет признаков с ненулевым IQR')
    scaled = (matrix.loc[:, usable].fillna(medians[usable]) - medians[usable]) / iqr[usable]
    fit = pd.DataFrame({'median': medians, 'iqr': iqr, 'included': usable})
    return scaled, mask, fit
