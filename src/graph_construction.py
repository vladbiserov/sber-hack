"""Этап 06: воспроизводимые неориентированные сети экономического сходства.

Вершины всегда соответствуют исходному порядку муниципалитетов.
Матрицы CSR имеют нулевую диагональ и симметричны.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from sklearn.neighbors import NearestNeighbors
from .feature_analysis import parse_dynamic_columns, window_view


def feature_at(frame: pd.DataFrame, mapping: dict, month: str | pd.Period,
               length: int = 1, representation: str = 'raw', zero_floor: float = 1e-6) -> pd.DataFrame:
    """Окно [t-length+1, t], только прошлое и текущий месяц."""
    return window_view(frame, mapping, month, length, representation, zero_floor)


def _validated_values(x: pd.DataFrame) -> np.ndarray:
    if not x.index.is_unique or x.index.hasnans:
        raise ValueError('Индекс должен быть уникальным и без пропусков')
    arr = x.to_numpy(dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2 or not np.isfinite(arr).all():
        raise ValueError('Ожидается числовая матрица без NaN/Inf')
    return arr


def directed_knn(x: pd.DataFrame, k: int, metric: str = 'euclidean') -> sparse.csr_matrix:
    """Бинарная направленная kNN (до k соседей без петель).

    Обрабатываем совпадающие векторы: исключаем собственную вершину по ID,
    а не предполагаем, что она всегда первая при нулевом расстоянии.
    """
    a = _validated_values(x)
    n = len(a)
    if not 1 <= k < n:
        raise ValueError('Должно выполняться 1 <= k < n')
    # Для 2016 территорий попарная матрица позволяет детерминированное разрешение ничьих.
    dist = cdist(a, a, metric=metric)
    np.fill_diagonal(dist, np.inf)
    order = np.argsort(dist, axis=1, kind='stable')[:, :k]
    row = np.repeat(np.arange(n), k)
    return sparse.csr_matrix((np.ones(n*k, dtype=np.int8), (row, order.ravel())), shape=(n, n))


def graph_from_knn(x: pd.DataFrame, k: int, mode: str = 'union',
                   bandwidth: str = 'median', metric: str = 'euclidean') -> tuple[sparse.csr_matrix, dict]:
    """Union kNN или mutual kNN; веса exp(-(d/sigma)^2).

    sigma = медиана строго положительных расстояний на выбранных рёбрах;
    если их нет, sigma=1 и веса совпадающих точек равны 1.
    """
    a = _validated_values(x)
    d = directed_knn(x, k, metric)
    binary = d.maximum(d.T) if mode == 'union' else d.minimum(d.T)
    if mode not in ('union','mutual'):
        raise ValueError('mode должен быть union или mutual')
    upper = sparse.triu(binary, k=1).tocoo()
    distances = np.linalg.norm(a[upper.row]-a[upper.col], axis=1) if metric == 'euclidean' else cdist(a, a, metric=metric)[upper.row,upper.col]
    positive = distances[distances > 0]
    if bandwidth != 'median':
        raise ValueError('Поддерживается bandwidth=median')
    sigma = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('Недопустимый bandwidth')
    weights = np.exp(-np.square(distances/sigma))
    out = sparse.coo_matrix((np.r_[weights, weights],
                            (np.r_[upper.row,upper.col],np.r_[upper.col,upper.row])),shape=(len(a),len(a))).tocsr()
    out.eliminate_zeros()
    return out, {'sigma':sigma, 'zero_distance_edges':int((distances==0).sum()), 'mode':mode,'k':k}


def graph_cosine_knn(x: pd.DataFrame, k: int, bandwidth: str='median') -> tuple[sparse.csr_matrix,dict]:
    """Альтернатива: kNN по косинусному расстоянию с той же весовой функцией.

    Для нулевых векторов cosine неопределён: случай останавливается явно.
    """
    a = _validated_values(x)
    if np.any(np.linalg.norm(a,axis=1) <= 1e-12):
        raise ValueError('Нулевая норма: косинусное расстояние не определено')
    return graph_from_knn(x, k, mode='union', bandwidth=bandwidth, metric='cosine')


def graph_metrics(adjacency: sparse.spmatrix) -> dict:
    """Метрики неориентированного графа без петель (считаем ребро однократно)."""
    a = adjacency.tocsr()
    n = a.shape[0]
    if a.shape[1] != n or (a-a.T).nnz or np.any(a.diagonal()!=0):
        raise ValueError('Матрица должна быть квадратной симметричной без петель')
    if a.nnz and (not np.isfinite(a.data).all() or (a.data<=0).any()):
        raise ValueError('Веса должны быть конечными положительными')
    components, labels = connected_components(a, directed=False)
    sizes = np.bincount(labels, minlength=components)
    deg = np.diff(a.indptr)
    w = sparse.triu(a,k=1).data
    edges = int(a.nnz//2)
    return {'nodes':n, 'edges':edges, 'density':2*edges/(n*(n-1)),
            'components':int(components), 'giant_share':float(sizes.max()/n),
            'degree_mean':float(deg.mean()), 'degree_p10':float(np.quantile(deg,.1)),
            'degree_p50':float(np.median(deg)), 'degree_p90':float(np.quantile(deg,.9)),
            'isolated_share':float(np.mean(deg==0)),
            'weight_p10':float(np.quantile(w,.1)) if edges else np.nan,
            'weight_p50':float(np.median(w)) if edges else np.nan,
            'weight_p90':float(np.quantile(w,.9)) if edges else np.nan}


def edge_jaccard(left: sparse.spmatrix, right: sparse.spmatrix) -> float:
    """Жаккар множеств неориентированных рёбер (без учёта весов)."""
    a = sparse.triu(left,k=1).tocsr().astype(bool)
    b = sparse.triu(right,k=1).tocsr().astype(bool)
    intersection = a.multiply(b).nnz
    union = a.nnz + b.nnz - intersection
    return float(intersection / union) if union else 1.0


def build_graph(x: pd.DataFrame, method: str, k: int) -> tuple[sparse.csr_matrix,dict]:
    if method == 'cosine':return graph_cosine_knn(x,k)
    if method in ('union','mutual'):return graph_from_knn(x,k,mode=method)
    raise ValueError('Неизвестный метод')


def experiment(frame: pd.DataFrame, mapping: dict, months: list[str],
               methods: list[str], ks: list[int], windows: list[int],
               representations: list[str], zero_floor: float=1e-6) -> pd.DataFrame:
    """Сравнение всех комбинаций на допустимых месяцах; ничего не записывает на диск.

    Для каждого варианта оценивается соседняя временная устойчивость (Jaccard).
    Результат — длинная таблица с метриками по месяцам.
    """
    records=[]
    for repr_name in representations:
        for length in windows:
            valid = [str(m) for m in months if all((pd.Period(m,freq='M')-i, 'share_food') in mapping for i in range(length))]
            features = {m:feature_at(frame,mapping,m,length,repr_name,zero_floor) for m in valid}
            for method in methods:
                for k in ks:
                    last=None
                    for month in valid:
                        adj,meta=build_graph(features[month],method,k)
                        rec={'month':month,'representation':repr_name,'window':length,'method':method,'k':k,
                             **graph_metrics(adj),'sigma':meta['sigma'],
                             'jaccard_prev':edge_jaccard(last,adj) if last is not None else np.nan}
                        records.append(rec)
                        last=adj
    return pd.DataFrame(records)


def save_sequence(frame: pd.DataFrame, mapping: dict, months: list[str],
                  directory: Path, *, method: str, k:int, representation:str,
                  window:int, zero_floor:float=1e-6, overwrite:bool=False) -> dict:
    """Сохранить только выбранный вариант в CSR .npz + метаданные JSON.

    Для окон >1 первые месяцы недоступны; запрещаем выдавать неполную
    последовательность за 24 графа. Для 24 графов используйте window=1,
    либо предоставьте историю до первого месяца.
    """
    directory=Path(directory)
    if window != 1:
        raise ValueError('Для полного ряда 24 графов требуется window=1 или дополнительная предыстория')
    if len(months)!=24:
        raise ValueError('Для сохранения ожидается 24 месяца')
    if directory.exists() and any(directory.iterdir()) and not overwrite:
        raise FileExistsError(f'Каталог уже содержит файлы: {directory}')
    directory.mkdir(parents=True,exist_ok=True)
    entries=[]
    for month in months:
        x=feature_at(frame,mapping,month,window,representation,zero_floor)
        a,meta=build_graph(x,method,k)
        filename=f'graph_{month}.npz'
        sparse.save_npz(directory/filename,a,compressed=True)
        entries.append({'month':month,'file':filename,'sigma':meta['sigma'], **graph_metrics(a)})
    identifiers=[str(v) for v in frame.index.tolist()]
    payload={'method':method,'k':k,'representation':representation,'window':window,
             'zero_floor':zero_floor,'node_ids':identifiers,'graphs':entries,
             'interpretation':'Экономическое сходство; не наблюдаемые потоки'}
    (directory/'manifest.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    return payload
