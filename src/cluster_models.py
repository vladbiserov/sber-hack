"""Этап 07: сравнение K-Means, spectral clustering и Leiden на фиксированном V.

Модуль не изменяет и не сохраняет исходные графы шага 06.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, silhouette_score

from .graph_construction import feature_at


def read_graph_sequence(directory: str | Path, index: pd.Index, months: list[str]) -> dict[str, sparse.csr_matrix]:
    """Чтение manifest.json и контроль строгого соответствия вершин/месяцев."""
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if not index.is_unique or index.hasnans:
        raise ValueError('Индекс муниципалитетов неуникален или содержит пропуски')
    if list(map(str, index)) != manifest['node_ids']:
        raise ValueError('Порядок или идентификаторы муниципалитетов не совпадают с manifest.json')
    listed = [str(row['month']) for row in manifest['graphs']]
    if listed != list(map(str, months)):
        raise ValueError('Месяцы или их порядок не совпадают с manifest.json')
    if manifest.get('representation') != 'clr' or manifest.get('window') != 1:
        raise ValueError('Графы должны соответствовать выбранному месячному CLR-пространству')
    result = {}
    n = len(index)
    for row in manifest['graphs']:
        path = directory / row['file']
        graph = sparse.load_npz(path).tocsr()
        graph.sum_duplicates()
        graph.eliminate_zeros()
        if graph.shape != (n, n) or (graph-graph.T).nnz or np.any(graph.diagonal() != 0):
            raise ValueError(f'Неверная структура графа {path}')
        if graph.data.size and (not np.isfinite(graph.data).all() or np.any(graph.data <= 0)):
            raise ValueError(f'Недопустимые веса {path}')
        result[str(row['month'])] = graph
    return result


def run_partition(method: str, features: pd.DataFrame, graph: sparse.csr_matrix,
                  parameter: float, seed: int) -> np.ndarray:
    """Получить метки без преобразования graph в плотную матрицу."""
    n = len(features)
    if graph.shape != (n, n):
        raise ValueError('Число вершин и строк признаков различается')
    if method == 'kmeans':
        labels = KMeans(n_clusters=int(parameter), random_state=seed, n_init=10).fit_predict(features)
    elif method == 'spectral':
        labels = SpectralClustering(n_clusters=int(parameter), affinity='precomputed',
                                    assign_labels='kmeans', random_state=seed,
                                    eigen_solver='arpack', n_init=10).fit_predict(graph)
    elif method == 'leiden':
        try:
            import igraph as ig
            import leidenalg
        except ImportError as exc:
            raise ImportError('Для Leiden установите: pip install python-igraph leidenalg') from exc
        upper = sparse.triu(graph, k=1).tocoo()
        network = ig.Graph(n=n, edges=list(zip(upper.row.tolist(), upper.col.tolist())), directed=False)
        network.es['weight'] = upper.data.astype(float).tolist()
        partition = leidenalg.find_partition(
            network, leidenalg.RBConfigurationVertexPartition,
            weights='weight', resolution_parameter=float(parameter), seed=int(seed), n_iterations=-1)
        labels = np.asarray(partition.membership, dtype=int)
    else:
        raise ValueError(f'Неизвестный алгоритм: {method}')
    if labels.shape != (n,) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError('Неправильные кластерные метки')
    return labels


def weighted_modularity(graph: sparse.csr_matrix, labels: np.ndarray) -> float:
    """Нормированная модулярность Ньюмана—Гирвана неориентированного графа."""
    labels = np.asarray(labels)
    strengths = np.asarray(graph.sum(axis=1)).ravel()
    total = float(strengths.sum())
    if total <= 0:
        return float('nan')
    groups, inverse = np.unique(labels, return_inverse=True)
    totals = np.bincount(inverse, weights=strengths, minlength=len(groups))
    coo = graph.tocoo()
    within = float(coo.data[inverse[coo.row] == inverse[coo.col]].sum())
    return float(within / total - np.square(totals / total).sum())


def partition_metrics(features: pd.DataFrame, graph: sparse.csr_matrix,
                      labels: np.ndarray, seed: int, sample_size: int = 500) -> dict:
    """Всем методам ставятся одни и те же оценки на месячных CLR-атрибутах.

    silhouette используется по фиксированной воспроизводимой подвыборке
    (одни и те же индексы при любом методе/seed для данного месяца).
    """
    values = features.to_numpy(dtype=float)
    labels = np.asarray(labels)
    n = len(labels)
    sizes = np.unique(labels, return_counts=True)[1]
    k = len(sizes)
    result = {'clusters': k, 'size_min': int(sizes.min()),
              'size_median': float(np.median(sizes)), 'size_max': int(sizes.max()),
              'largest_share': float(sizes.max()/n),
              'modularity': weighted_modularity(graph, labels)}
    result['calinski_harabasz'] = float(calinski_harabasz_score(values, labels)) if 1 < k < n else float('nan')
    rng = np.random.default_rng(2026)
    sampled = np.sort(rng.choice(n, min(sample_size, n), replace=False))
    subset = labels[sampled]
    if 1 < len(np.unique(subset)) < len(sampled):
        result['silhouette'] = float(silhouette_score(values[sampled], subset, metric='euclidean'))
    else:
        result['silhouette'] = float('nan')
    return result


def evaluate_grid(frame: pd.DataFrame, mapping: dict, months: list[str],
                  graphs: dict[str, sparse.csr_matrix], candidates: dict[str, list],
                  seeds: list[int], zero_floor: float = 1e-6, sample_size: int = 500,
                  keep_labels: bool = False):
    """Компактная таблица метрик и, при запросе, метки каждого запуска."""
    rows, labels_store = [], {}
    for month in months:
        X = feature_at(frame, mapping, month, 1, 'clr', zero_floor)
        A = graphs[month]
        for method, parameters in candidates.items():
            for parameter in parameters:
                for seed in seeds:
                    labels = run_partition(method, X, A, parameter, seed)
                    metrics = partition_metrics(X, A, labels, seed, sample_size)
                    rows.append({'month': str(month), 'method': method,
                                 'parameter': float(parameter), 'seed': int(seed), **metrics})
                    if keep_labels:
                        labels_store[(str(month), method, float(parameter), int(seed))] = labels
    table = pd.DataFrame(rows)
    return table, labels_store


def add_repeatability(metrics: pd.DataFrame, labels_store: dict) -> pd.DataFrame:
    """ARI между всеми парами seed для каждого месяца/метода/параметра."""
    import itertools
    rows = []
    for (month, method, parameter), group in metrics.groupby(['month', 'method', 'parameter'], sort=False):
        runs = [labels_store[(month, method, float(parameter), int(s))] for s in group.seed]
        scores = [adjusted_rand_score(a, b) for a, b in itertools.combinations(runs, 2)]
        rows.append({'month': month, 'method': method, 'parameter': parameter,
                     'ari_repeats': float(np.mean(scores)) if scores else np.nan})
    return metrics.merge(pd.DataFrame(rows), on=['month', 'method', 'parameter'], validate='many_to_one')


def selected_labels(labels_store: dict, months: list[str], selected: dict,
                    seeds: list[int], index: pd.Index) -> pd.DataFrame:
    """По одному воспроизводимому запуску каждого метода и месяца, общий индекс."""
    columns = {}
    for month in months:
        for method, parameter in selected.items():
            key = (str(month), method, float(parameter), int(seeds[0]))
            columns[(str(month), method)] = labels_store[key]
    out = pd.DataFrame(columns, index=index)
    out.columns = pd.MultiIndex.from_tuples(out.columns, names=['month', 'method'])
    return out


def export_results(directory: Path, labels: pd.DataFrame, metrics: pd.DataFrame,
                   config: dict, selected: dict):
    """Только метки, таблица метрик и компактная информация о параметрах."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(directory / 'labels.parquet')
    metrics.to_parquet(directory / 'metrics.parquet', index=False)
    metadata = {'selected_parameters': selected,
                'seeds': config['seeds'], 'source_graph_directory': config['graph_directory'],
                'feature_path': config['feature_path'],
                'silhouette_sample_size': config['silhouette_sample_size'],
                'selection_basis': 'контрольные месяцы; ретроспективная настройка; не прогноз'}
    (directory / 'experiment.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
