"""Этап 09: геометрические и сетевые индексы сохранённых разбиений.

AVI/AVU: Shalileh et al. (2025), формулы (18)–(21), бинарная смежность.
MQ: принятая в проекте трактовка weighted Newman–Girvan modularity Q.
Официальное конкурсное определение MQ отдельно не подтверждено.
"""
from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             adjusted_rand_score, normalized_mutual_info_score)

from .cluster_models import weighted_modularity
from .graph_construction import feature_at


def check_partition(X: pd.DataFrame, A: sparse.spmatrix, labels: Sequence) -> tuple[np.ndarray, np.ndarray]:
    """Строгое выравнивание, отсутствие NaN, конечность CLR, корректность графа."""
    n = len(X)
    if not X.index.is_unique or X.index.hasnans:
        raise ValueError('Невалидный индекс муниципалитетов')
    if isinstance(labels, pd.Series) and not labels.index.equals(X.index):
        raise ValueError('Порядок индекса меток не совпадает с признаками')
    arr = np.asarray(labels)
    if arr.shape != (n,) or pd.isna(arr).any():
        raise ValueError('Метки должны быть вектором длины n без пропусков')
    values = X.to_numpy(dtype=float)
    if values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError('Неполные либо неконечные признаки')
    if A.shape != (n, n):
        raise ValueError('Размер графа не совпадает с числом муниципалитетов')
    graph = A.tocsr()
    if (graph - graph.T).nnz or np.any(graph.diagonal() != 0):
        raise ValueError('Граф неориентированного сходства должен быть симметричным без петель')
    if (not np.isfinite(graph.data).all()) or (graph.data < 0).any():
        raise ValueError('Некорректные веса графа')
    return values, arr


def s_dbw_halkidi(values: np.ndarray, labels: Sequence) -> float:
    """S_Dbw = Scatt + Dens_bw, вариант Halkidi (евклидово расстояние).

    Стандартное отклонение точек вычисляется с ddof=0; радиус плотности:
    (1/K) sqrt(sum_k ||variance(C_k)||_2). На середине центров плотность
    измеряется по объединению двух кластеров. При вырождении -> NaN.
    """
    X = np.asarray(values, dtype=float)
    y = np.asarray(labels)
    if X.ndim != 2 or y.shape != (len(X),) or not np.isfinite(X).all() or pd.isna(y).any():
        raise ValueError('Некорректные точки или метки')
    groups, inv = np.unique(y, return_inverse=True)
    k, n = len(groups), len(X)
    if not 2 <= k < n:
        return float('nan')
    var_all = np.var(X, axis=0)
    norm_all = float(np.linalg.norm(var_all))
    if not np.isfinite(norm_all) or norm_all <= 0:
        return float('nan')
    chunks = [X[inv == a] for a in range(k)]
    centers = np.vstack([p.mean(axis=0) for p in chunks])
    norms = np.array([np.linalg.norm(np.var(p, axis=0)) for p in chunks])
    scatt = float(norms.sum() / (k * norm_all))
    radius = float(np.sqrt(norms.sum()) / k)
    if radius <= 0 or not np.isfinite(radius):
        return float('nan')
    central_dens = [int(np.count_nonzero(np.linalg.norm(p - centers[a], axis=1) <= radius))
                    for a, p in enumerate(chunks)]
    ratio_sum = 0.0
    for a in range(k):
        for b in range(a + 1, k):
            mid = 0.5 * (centers[a] + centers[b])
            joint = np.concatenate((chunks[a], chunks[b]), axis=0)
            between = int(np.count_nonzero(np.linalg.norm(joint - mid, axis=1) <= radius))
            denom = max(central_dens[a], central_dens[b])
            if denom <= 0:
                return float('nan')
            ratio_sum += between / denom
    return float(scatt + 2.0 * ratio_sum / (k * (k - 1)))


def evaluate_partition(features: pd.DataFrame, graph: sparse.spmatrix,
                       labels: Sequence) -> dict:
    """Все геометрические метрики на 2016 объектах, без случайной подвыборки."""
    X, y = check_partition(features, graph, labels)
    _, sizes = np.unique(y, return_counts=True)
    n, k = len(y), len(sizes)
    valid = 2 <= k < n
    result = {
        'n_nodes': n,
        'n_clusters': k,
        'min_cluster_size': int(sizes.min()),
        'largest_cluster_share': float(sizes.max() / n),
        'singletons': int(np.count_nonzero(sizes == 1)),
        'sw': float('nan'), 'ch': float('nan'), 's_dbw': float('nan'),
        'modularity': float(weighted_modularity(graph.tocsr(), y)),
        'status': 'ok' if valid else 'degenerate_partition',
    }
    if valid:
        within_ss = sum(float(np.square(X[y == lab] - X[y == lab].mean(axis=0)).sum())
                        for lab in np.unique(y))
        if within_ss <= np.finfo(float).eps:
            result['status'] = 'zero_within_variance'
            return result
        result['sw'] = float(silhouette_score(X, y, metric='euclidean'))
        result['ch'] = float(calinski_harabasz_score(X, y))
        result['s_dbw'] = s_dbw_halkidi(X, y)
        if not np.isfinite(result['s_dbw']):
            result['status'] = 's_dbw_undefined'
    return result


def load_saved_partitions(stage07_dir: str | Path, stage08_dir: str | Path,
                          index: pd.Index, months: Sequence[str],
                          include_temporal: bool = True) -> tuple[dict[str, pd.DataFrame], dict]:
    """Загрузить только сохранённые метки; 08 опционален и строго валидируется."""
    path07 = Path(stage07_dir)
    labels07 = pd.read_parquet(path07 / 'labels.parquet')
    if not labels07.index.equals(index):
        raise ValueError('Метки шага 07 не совпадают с индексом X_dynamic')
    if not isinstance(labels07.columns, pd.MultiIndex) or labels07.columns.nlevels != 2:
        raise ValueError('Ожидается MultiIndex (month,method) шага 07')
    out = {}
    for method in ['kmeans', 'spectral', 'leiden']:
        cols = [(str(month), method) for month in months]
        if any(key not in labels07.columns for key in cols):
            raise ValueError(f'Нет полного набора меток шага 07 для {method}')
        part = pd.DataFrame({str(m):labels07[(str(m), method)] for m in months}, index=index)
        if part.isna().any().any():
            raise ValueError(f'Пропуски в разбиении {method}')
        out[method] = part
    meta = {'temporal_status': 'not_requested', 'temporal_selected_lambda': None}
    if include_temporal:
        d = Path(stage08_dir)
        labelsfile = d / 'temporal_labels.parquet'
        metafile = d / 'temporal_experiment.json'
        if labelsfile.exists() != metafile.exists():
            raise FileNotFoundError('Неполный комплект этапа 08: метки и метаданные должны существовать вместе')
        if labelsfile.exists():
            experiment = json.loads(metafile.read_text(encoding='utf-8'))
            if 'selected_lambda' not in experiment or not np.isfinite(float(experiment['selected_lambda'])):
                raise ValueError('В метаданных 08 отсутствует подтверждённая lambda')
            labels08 = pd.read_parquet(labelsfile)
            if not labels08.index.equals(index) or not isinstance(labels08.columns, pd.MultiIndex):
                raise ValueError('Неверный индекс или колонки этапа 08')
            cols = [(str(m), 'temporal') for m in months]
            if any(key not in labels08.columns for key in cols):
                raise ValueError('Отсутствуют месячные метки временной модели')
            temporal = pd.DataFrame({str(m): labels08[(str(m),'temporal')] for m in months}, index=index)
            if temporal.isna().any().any():
                raise ValueError('Пропуски в метках временной модели')
            out['temporal_kmeans'] = temporal
            meta = {'temporal_status':'loaded', 'temporal_selected_lambda':float(experiment['selected_lambda'])}
        else:
            meta = {'temporal_status':'not_saved', 'temporal_selected_lambda':None}
    return out, meta


def evaluate_all(frame: pd.DataFrame, mapping: dict, months: Sequence[str],
                 graphs: Mapping[str, sparse.spmatrix], partitions: Mapping[str, pd.DataFrame],
                 zero_floor: float, method_params: Mapping[str, float] | None = None,
                 temporal_lambda: float | None = None) -> pd.DataFrame:
    """Месячная панель показателей; ARI/NMI соседних разбиений инвариантны к ID."""
    months = [str(m) for m in months]
    if set(graphs) != set(months):
        raise ValueError('Графы покрывают не все месяцы')
    params = method_params or {}
    for method, labels in partitions.items():
        if not labels.index.equals(frame.index) or list(labels.columns) != months:
            raise ValueError(f'Неправильный индекс или временной порядок для {method}')
    rows = []
    for t, month in enumerate(months):
        X = feature_at(frame, mapping, month, 1, 'clr', zero_floor)
        if not X.index.equals(frame.index):
            raise ValueError('Признаки изменили порядок муниципалитетов')
        A = graphs[month]
        for method, labels in partitions.items():
            y = labels[month]
            metrics = evaluate_partition(X, A, y)
            row = {'month':month, 'method':method,
                   'configuration_id': ('temporal_lambda_' + str(temporal_lambda) if method=='temporal_kmeans'
                                        else f'{method}_parameter_{params.get(method, "unknown")}'),
                   'parameter':float(temporal_lambda if method == 'temporal_kmeans' and temporal_lambda is not None
                                     else params.get(method, np.nan)),
                   'feature_space':'CLR_6_monthly', 'graph_space':'CLR_union_kNN_k10',
                   **metrics, 'ari_prev':np.nan, 'nmi_prev':np.nan}
            if t > 0:
                before = labels[months[t-1]].to_numpy()
                current = y.to_numpy()
                row['ari_prev'] = float(adjusted_rand_score(before, current))
                row['nmi_prev'] = float(normalized_mutual_info_score(before, current))
            rows.append(row)
    return pd.DataFrame(rows)


def network_quality_indices(graph: sparse.spmatrix, labels: Sequence) -> dict[str, float]:
    """AVI/AVU публикации Shalileh et al., (18)–(21), и weighted modularity Q.

    Бинаризация применяется ТОЛЬКО к AVI/AVU, MQ использует исходные веса.
    Если какая-либо доля не определена (нулевой знаменатель), возвращается NaN.
    Не поддерживает петли, ориентированные/отрицательные графы и пустой граф.
    """
    A = sparse.csr_matrix(graph)
    y = np.asarray(labels)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or y.shape != (A.shape[0],):
        raise ValueError('Размер графа и меток не совпадает')
    if pd.isna(y).any():
        raise ValueError('Пропуски в кластерных метках')
    A = A.copy()
    A.eliminate_zeros()
    if (A - A.T).nnz or np.any(A.diagonal() != 0):
        raise ValueError('Нужен симметричный граф без петель')
    if not np.isfinite(A.data).all() or np.any(A.data < 0):
        raise ValueError('Веса должны быть конечными и неотрицательными')
    _, groups = np.unique(y, return_inverse=True)
    k = int(groups.max()) + 1 if len(groups) else 0
    mq = float(weighted_modularity(A, y))
    if k < 2 or A.nnz == 0:
        return {'avi': float('nan'), 'avu': float('nan'), 'mq': mq}

    binary = A.copy()
    binary.data[:] = 1.0
    membership = sparse.csr_matrix((np.ones(len(groups)),
                                    (np.arange(len(groups)), groups)),
                                   shape=(len(groups), k))
    B = (membership.T @ binary @ membership).toarray()
    internal = B.diagonal().copy()
    external = B.sum(axis=1) - internal
    degree_sum = internal + external
    avi = float(np.mean(internal / degree_sum)) if np.all(degree_sum > 0) else float('nan')

    denominator = external[:, None] + external[None, :] - B
    off_diagonal = ~np.eye(k, dtype=bool)
    valid = denominator[off_diagonal] > 0
    avu = (float(np.sum((B / np.where(denominator > 0, denominator, 1.0))[off_diagonal]) / k)
           if valid.all() else float('nan'))
    return {'avi': avi, 'avu': avu, 'mq': mq}


def add_network_indices(table: pd.DataFrame,
                        graphs: Mapping[str, sparse.spmatrix],
                        partitions: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Досчитать только сетевые ICVI для уже оценённых месячных разбиений."""
    if table.duplicated(['month', 'method']).any():
        raise ValueError('Дубли пар (month, method) в таблице')
    out = table.copy()
    for key in ('avi', 'avu', 'mq'):
        out[key] = np.nan
    for row in out.itertuples():
        month, method = str(row.month), str(row.method)
        if month not in graphs or method not in partitions:
            raise KeyError(f'Нет графа/меток: {month}, {method}')
        block = partitions[method]
        if month not in block or len(block) != graphs[month].shape[0]:
            raise ValueError(f'Некорректная размерность или месяц: {method}, {month}')
        vals = network_quality_indices(graphs[month], block[month].to_numpy())
        if 'modularity' in out.columns and not np.isclose(
            vals['mq'], float(row.modularity), atol=1e-10, rtol=1e-10, equal_nan=True
        ):
            raise ValueError(f'MQ и исходная Modularity расходятся: {method}, {month}')
        for key, val in vals.items():
            out.at[row.Index, key] = val
    return out


def merge_repeatability(table: pd.DataFrame, stage07_dir: str | Path,
                        method_params: Mapping[str, float]) -> pd.DataFrame:
    """Если шаг 07 сохранил ARI повторов — добавить, не обучая повторно."""
    table = table.copy()
    path = Path(stage07_dir) / 'metrics.parquet'
    table['ari_repeats'] = np.nan
    if not path.exists():
        return table
    prior = pd.read_parquet(path)
    need = {'month','method','parameter','ari_repeats'}
    if not need.issubset(prior.columns):
        return table
    for method, parameter in method_params.items():
        block = prior.loc[(prior.method==method)&np.isclose(prior.parameter.astype(float),float(parameter))]
        reference = block.groupby('month')['ari_repeats'].median()
        mask = table.method == method
        table.loc[mask,'ari_repeats'] = table.loc[mask,'month'].map(reference).to_numpy(dtype=float)
    return table


def monthly_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Медианы и IQR, отдельно лучшие/худшие даты для двух направлений ICVI."""
    columns=['sw','ch','s_dbw','modularity','ari_prev','nmi_prev','n_clusters','largest_cluster_share']
    columns += [c for c in ('avi', 'avu', 'mq') if c in table.columns]
    rows=[]
    for method, part in table.groupby('method', sort=False):
        rec={'method':method,'months':part.month.nunique(),'invalid_months':int((part.status!='ok').sum())}
        for col in columns:
            vals=part[col].dropna()
            rec[col+'_median']=float(vals.median()) if len(vals) else np.nan
            rec[col+'_iqr']=float(vals.quantile(.75)-vals.quantile(.25)) if len(vals) else np.nan
            if col in ('avi','avu','mq'):
                rec[col+'_n_defined']=int(len(vals))
        direction=[('sw',True),('ch',True),('s_dbw',False),('modularity',True)]
        direction += [(c, maximize) for c,maximize in
                      [('avi',True),('avu',False),('mq',True)] if c in table.columns]
        for col, best_high in direction:
            valid=part.dropna(subset=[col]);
            if len(valid):
                rec[col+'_best_month']=str(valid.loc[valid[col].idxmax() if best_high else valid[col].idxmin(),'month'])
                rec[col+'_worst_month']=str(valid.loc[valid[col].idxmin() if best_high else valid[col].idxmax(),'month'])
            else:
                rec[col+'_best_month']=None;rec[col+'_worst_month']=None
        rows.append(rec)
    return pd.DataFrame(rows).set_index('method')


def save_metrics(path: str | Path, metrics: pd.DataFrame) -> Path:
    """Одна таблица без повторного сохранения графов/меток."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(out, index=False)
    return out


def plot_quality_pairs(table: pd.DataFrame, left: str, right: str,
                       labels: tuple[str, str], *, figsize: tuple = (12, 4.0)):
    """Один компактный график с двумя рядами метрик, без усреднения их шкал."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    methods = list(table.method.unique())
    for ax, key, title in zip(axes, (left, right), labels):
        for method in methods:
            part = table.loc[table.method == method].sort_values('month')
            ax.plot(part['month'], part[key], marker='.', markersize=3,
                    linewidth=1.2, label=method)
        ax.set_title(title)
        ax.set_xticks(table.month.unique()[::4]); ax.tick_params(axis='x',rotation=30)
        ax.grid(alpha=.2)
    axes[0].legend(loc='best', fontsize=8)
    return fig, axes
