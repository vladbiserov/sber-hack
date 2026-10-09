"""Этап 08: ретроспективное сопоставление и временная регуляризация кластеров.

Идентификаторы муниципалитетов сохраняются без перестановки. Временные
связи соединяют только копии ОДНОГО муниципалитета в соседних месяцах.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .cluster_models import weighted_modularity
from .graph_construction import feature_at


def validate_inputs(index: pd.Index, labels: pd.DataFrame,
                    graphs: Mapping[str, sparse.spmatrix], months: Sequence[str],
                    method: str) -> None:
    """Строго проверить идентичность V, состав месяцев и допустимость меток."""
    months = [str(m) for m in months]
    if len(months) != 24 or len(set(months)) != len(months):
        raise ValueError('Ожидаются 24 различных месяца')
    periods = pd.PeriodIndex(months, freq='M')
    if not periods.is_monotonic_increasing or any(b - a != 1 for a, b in zip(periods.asi8[:-1], periods.asi8[1:])):
        raise ValueError('Месяцы должны быть последовательными и возрастающими')
    if not index.is_unique or index.hasnans or not labels.index.equals(index):
        raise ValueError('Идентификаторы и порядок муниципалитетов не совпадают')
    if not isinstance(labels.columns, pd.MultiIndex) or labels.columns.nlevels != 2:
        raise ValueError('Ожидается labels.parquet с колонками (month, method)')
    missing = [(m, method) for m in months if (m, method) not in labels.columns]
    if missing:
        raise KeyError(f'Отсутствуют метки: {missing[:3]}')
    if set(graphs) != set(months):
        raise ValueError('Месяцы графов и меток не совпадают')
    for m in months:
        lab = labels[(m, method)]
        if lab.isna().any() or not pd.api.types.is_integer_dtype(lab):
            raise ValueError(f'Невалидные метки в {m}')
        g = graphs[m].tocsr()
        if g.shape != (len(index), len(index)) or (g - g.T).nnz or np.any(g.diagonal() != 0):
            raise ValueError(f'Невалидный граф для {m}')


def raw_labels(labels: pd.DataFrame, months: Sequence[str], method: str) -> pd.DataFrame:
    """Извлечь месячные метки метода без изменения индекса."""
    return pd.DataFrame({str(m): labels[(str(m), method)].to_numpy(dtype=np.int64)
                         for m in months}, index=labels.index)


def transition_matrix(left: Sequence[int], right: Sequence[int]) -> pd.DataFrame:
    """Таблица численностей n_ab с отсортированными локальными идентификаторами."""
    a, b = np.asarray(left), np.asarray(right)
    if a.ndim != 1 or b.ndim != 1 or len(a) != len(b) or len(a) == 0:
        raise ValueError('Ожидаются непустые одинаковые векторы меток')
    if pd.isna(a).any() or pd.isna(b).any():
        raise ValueError('Пропущены кластерные метки')
    source, ai = np.unique(a, return_inverse=True)
    target, bi = np.unique(b, return_inverse=True)
    counts = np.zeros((len(source), len(target)), dtype=np.int64)
    np.add.at(counts, (ai, bi), 1)
    return pd.DataFrame(counts, index=source, columns=target)


def align_history(raw: pd.DataFrame, min_jaccard: float = 0.0,
                  event_fraction: float = 0.1, min_event_size: int = 5
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Восстановить сквозные ID по Hungarian, не смешивая splits/merges.

    Сначала максимизируется суммарное пересечение сообществ (Hungarian).
    Для каждой пары принимается соответствие только при положительном
    пересечении и Jaccard >= min_jaccard; прочим даются новые ID.
    Разделения и слияния регистрируются отдельно по ПОЛНОЙ матрице потоков.
    """
    if not 0 <= min_jaccard <= 1 or not 0 <= event_fraction <= 1 or min_event_size < 1:
        raise ValueError('Неверные пороги сопоставления')
    if not raw.index.is_unique or raw.index.hasnans or raw.isna().any().any():
        raise ValueError('Невалидные метки или индекс')
    months = list(raw.columns)
    if not months:
        raise ValueError('Нет месяцев')
    first = raw[months[0]].to_numpy()
    unique, count = np.unique(first, return_counts=True)
    order = unique[np.lexsort((unique, -count))]
    remap = {v: i for i, v in enumerate(order)}
    matched = {months[0]: np.asarray([remap[v] for v in first], dtype=np.int64)}
    next_id = len(unique)
    event_rows = []
    for previous, current in zip(months[:-1], months[1:]):
        before, after = matched[previous], raw[current].to_numpy()
        tab = transition_matrix(before, after)
        matrix = tab.to_numpy()
        row_sizes = matrix.sum(axis=1)
        col_sizes = matrix.sum(axis=0)
        ii, jj = linear_sum_assignment(-matrix)  # общий максимум пересечений
        mapping = {}
        for i, j in zip(ii, jj):
            overlap = int(matrix[i, j])
            union = int(row_sizes[i] + col_sizes[j] - overlap)
            jac = overlap / union if union else 0.0
            if overlap > 0 and jac >= min_jaccard:
                mapping[tab.columns[j]] = int(tab.index[i])
        for label in tab.columns:
            if label not in mapping:
                mapping[label] = next_id
                next_id += 1
        matched[current] = np.asarray([mapping[label] for label in after], dtype=np.int64)
        # Разделение/слияние определяются многими существенными потоками,
        # даже если Hungarian установил только одно главное соответствие.
        significant = (matrix >= min_event_size) & (
            matrix / row_sizes[:, None] >= event_fraction) & (
            matrix / col_sizes[None, :] >= event_fraction)
        for i in np.flatnonzero(significant.sum(axis=1) > 1):
            event_rows.append({'month_from': previous, 'month_to': current,
                               'event': 'split', 'cluster': int(tab.index[i]),
                               'branches': int(significant[i].sum())})
        for j in np.flatnonzero(significant.sum(axis=0) > 1):
            event_rows.append({'month_from': previous, 'month_to': current,
                               'event': 'merge', 'cluster': int(mapping[tab.columns[j]]),
                               'branches': int(significant[:, j].sum())})
    aligned = pd.DataFrame(matched, index=raw.index)
    events = pd.DataFrame(event_rows, columns=['month_from','month_to','event','cluster','branches'])
    return aligned, events


def transitions(aligned: pd.DataFrame) -> pd.DataFrame:
    """23 таблицы n_ab и P_ab=n_ab/n_a, в длинном формате (включая нули)."""
    rows = []
    months = list(aligned.columns)
    for left, right in zip(months[:-1], months[1:]):
        tab = transition_matrix(aligned[left], aligned[right])
        totals = tab.sum(axis=1)
        for a in tab.index:
            for b in tab.columns:
                n = int(tab.loc[a, b])
                rows.append({'month_from': left, 'month_to': right,
                             'source': int(a), 'target': int(b), 'count': n,
                             'source_size': int(totals.loc[a]),
                             'probability': float(n/totals.loc[a])})
    return pd.DataFrame(rows)


def time_metrics(aligned: pd.DataFrame, stable_jaccard: float = 0.5) -> pd.DataFrame:
    """Доля сохранивших метку, ARI/NMI, устойчивые сообщества."""
    if not 0 < stable_jaccard <= 1:
        raise ValueError('stable_jaccard должен быть в (0,1]')
    months = list(aligned.columns)
    rows = []
    for left, right in zip(months[:-1], months[1:]):
        a = aligned[left].to_numpy()
        b = aligned[right].to_numpy()
        tab = transition_matrix(a, b)
        stable = 0
        # устойчивое сообщество: тот же ID и Jaccard его состава >= порога
        for cluster in set(tab.index).intersection(tab.columns):
            n = int(tab.loc[cluster, cluster])
            union = int(tab.loc[cluster].sum() + tab[cluster].sum() - n)
            if union and n / union >= stable_jaccard:
                stable += 1
        rows.append({'month_from': left, 'month_to': right,
                     'cluster_count_from': int(len(np.unique(a))),
                     'cluster_count_to': int(len(np.unique(b))),
                     'stay_share': float(np.mean(a == b)),
                     'switch_share': float(np.mean(a != b)),
                     'ari': float(adjusted_rand_score(a, b)),
                     'nmi': float(normalized_mutual_info_score(a, b)),
                     'stable_communities': stable})
    return pd.DataFrame(rows)


def features_by_month(frame: pd.DataFrame, mapping: dict,
                      months: Sequence[str], zero_floor: float) -> dict[str, pd.DataFrame]:
    """Загрузить CLR-признаки точно за текущий месяц, без будущих данных."""
    result = {m: feature_at(frame, mapping, m, 1, 'clr', zero_floor) for m in months}
    if any(not x.index.equals(frame.index) for x in result.values()):
        raise ValueError('Потерян порядок муниципалитетов')
    return result


def attribute_change_association(aligned: pd.DataFrame, features: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Описательная связь смены метки с величиной месячного изменения CLR."""
    rows = []
    for previous, current in zip(aligned.columns[:-1], aligned.columns[1:]):
        x = features[previous].to_numpy(float)
        y = features[current].to_numpy(float)
        magnitudes = np.linalg.norm(y-x, axis=1)
        switched = aligned[previous].to_numpy() != aligned[current].to_numpy()
        stayed = ~switched
        correlation = (float(spearmanr(magnitudes, switched.astype(float)).statistic)
                       if switched.any() and stayed.any() and np.std(magnitudes) > 0 else np.nan)
        rows.append({'month_from': previous, 'month_to': current,
                     'n_switch': int(switched.sum()),
                     'median_clr_change_switch': float(np.median(magnitudes[switched])) if switched.any() else np.nan,
                     'median_clr_change_stay': float(np.median(magnitudes[stayed])) if stayed.any() else np.nan,
                     'spearman_change_vs_switch': correlation})
    return pd.DataFrame(rows)


def _reference_emissions(reference: pd.DataFrame, features: Mapping[str, pd.DataFrame],
                         graphs: Mapping[str, sparse.spmatrix], graph_beta: float
                        ) -> tuple[np.ndarray, np.ndarray, float]:
    """Привязанные к меткам 07 центроиды + опциональная опора на соседей графа.

    При graph_beta > 0 использованы ФИКСИРОВАННЫЕ метки соседей из 07,
    а не метки оптимизируемых состояний. Это явное приближение, позволяющее
    точно решить временную часть задачи алгоритмом Витерби.
    """
    if graph_beta < 0:
        raise ValueError('graph_beta неотрицателен')
    months = list(reference.columns)
    n, T = reference.shape
    names = np.unique(reference.to_numpy())
    K = len(names)
    if K < 2:
        raise ValueError('Модель требует как минимум два кластера')
    if any(set(reference[m].unique()) != set(names) for m in months):
        raise ValueError('Временная модель требует один и тот же набор K кластеров во все месяцы')
    states = {v: idx for idx, v in enumerate(names)}
    base = np.zeros((T,n,K), dtype=np.float64)
    assigned = []
    for t, month in enumerate(months):
        X = features[month].to_numpy(dtype=float)
        lab = reference[month].to_numpy()
        centers = np.vstack([X[lab == k].mean(axis=0) for k in names])
        base[t] = np.square(X[:,None,:]-centers[None,:,:]).sum(axis=2)
        assigned.extend(base[t, np.arange(n), [states[v] for v in lab]].tolist())
    errors = np.asarray(assigned)
    positive = errors[errors > 1e-14]
    scale = float(np.median(positive)) if positive.size else 1.0
    emission = base / scale
    if graph_beta:
        for t, month in enumerate(months):
            A = graphs[month].tocsr()
            lab = reference[month].to_numpy()
            indicator = np.column_stack([(lab == k).astype(float) for k in names])
            strengths = np.asarray(A.sum(axis=1)).ravel()
            fractions = np.divide(A @ indicator, strengths[:,None],
                                  out=np.zeros((n,K)), where=strengths[:,None] > 0)
            emission[t] += graph_beta * (1.0 - fractions)
    return emission, names, scale


def temporal_viterbi(emissions: np.ndarray, lambda_value: float,
                     index: pd.Index, months: Sequence[str], state_ids: np.ndarray
                    ) -> pd.DataFrame:
    """Точное решение Potts-связи между копиями муниципалитета в t и t+1.

    min sum_t emission[t,i,z_i,t] + lambda sum_t 1[z_i,t != z_i,t-1].
    Никаких рёбер между разными муниципалитетами ВО ВРЕМЕНИ не создаётся.
    """
    if not np.isfinite(lambda_value) or lambda_value < 0:
        raise ValueError('lambda должен быть неотрицательным')
    em = np.asarray(emissions, dtype=float)
    if em.ndim != 3 or not np.isfinite(em).all():
        raise ValueError('emissions ожидается конечной матрицей T × n × K')
    T, n, K = em.shape
    if n != len(index) or T != len(months) or K != len(state_ids):
        raise ValueError('Размеры emissions и индексов не совпадают')
    dp = em[0].copy()
    pointers = np.zeros((T,n,K), dtype=np.int32)
    penalty = lambda_value * (np.arange(K)[:,None] != np.arange(K)[None,:])
    for t in range(1,T):
        choices = dp[:,:,None] + penalty[None,:,:]  # n × previous × current
        pointers[t] = choices.argmin(axis=1)
        dp = em[t] + choices.min(axis=1)
    labels = np.empty((n,T), dtype=np.int32)
    labels[:,-1] = dp.argmin(axis=1)
    row = np.arange(n)
    for t in range(T-1,0,-1):
        labels[:,t-1] = pointers[t,row,labels[:,t]]
    return pd.DataFrame(state_ids[labels], index=index, columns=list(months))


def run_lambda_grid(reference: pd.DataFrame, features: Mapping[str, pd.DataFrame],
                    graphs: Mapping[str, sparse.spmatrix], lambda_values: Sequence[float],
                    graph_beta: float = 0.0
                    ) -> tuple[pd.DataFrame, dict[float, pd.DataFrame], dict]:
    """Единые эмиссии для λ, без переобучения центроидов под конкретное λ."""
    lambdas = [float(v) for v in lambda_values]
    if 0.0 not in lambdas or len(lambdas) != len(set(lambdas)) or any(l < 0 for l in lambdas):
        raise ValueError('В сетке нужен один контроль lambda=0 и неотрицательные уникальные значения')
    em, state_ids, scale = _reference_emissions(reference, features, graphs, graph_beta)
    encoded = {v:i for i,v in enumerate(state_ids)}
    T,n,K = em.shape
    rows, paths = [], {}
    baseline_encoded = np.column_stack([reference[m].map(encoded).to_numpy(int) for m in reference.columns])
    baseline_cost = float(em[np.arange(T)[:,None], np.arange(n)[None,:], baseline_encoded.T].mean())
    for lam in lambdas:
        labels = temporal_viterbi(em, lam, reference.index, list(reference.columns), state_ids)
        paths[lam] = labels
        temporal = time_metrics(labels)
        encoded_labels = np.column_stack([labels[m].map(encoded).to_numpy(int) for m in labels.columns])
        emission_loss = float(em[np.arange(T)[:,None], np.arange(n)[None,:], encoded_labels.T].mean())
        agreement = []
        modularity = []
        for month in reference.columns:
            A = graphs[month].tocsr()
            lab = labels[month].to_numpy()
            coo = sparse.triu(A,k=1).tocoo()
            agreement.append(float(coo.data[lab[coo.row] == lab[coo.col]].sum() / coo.data.sum())
                             if len(coo.data) else np.nan)
            modularity.append(weighted_modularity(A, lab))
        rows.append({'lambda':lam, 'graph_beta':graph_beta,
                     'stay_share_median':float(temporal.stay_share.median()),
                     'switch_share_median':float(temporal.switch_share.median()),
                     'ari_adjacent_median':float(temporal.ari.median()),
                     'nmi_adjacent_median':float(temporal.nmi.median()),
                     'clusters_median':float(np.median([labels[m].nunique() for m in labels])),
                     'clusters_min':int(min(labels[m].nunique() for m in labels)),
                     'modularity_median':float(np.nanmedian(modularity)),
                     'within_edge_weight_share_median':float(np.nanmedian(agreement)),
                     'ari_to_independent_median':float(np.median([
                         adjusted_rand_score(reference[m], labels[m]) for m in labels])),
                     'emission_cost':emission_loss,
                     'emission_cost_vs_reference':float(emission_loss / baseline_cost) if baseline_cost > 0 else np.nan})
    return pd.DataFrame(rows), paths, {'emission_scale':scale,
                                       'centroids': 'fixed_from_stage_07_monthly_partitions',
                                       'graph_beta':graph_beta}


def delta_transition_associations(aligned: pd.DataFrame, delta: pd.DataFrame,
                                 year: int = 2024) -> pd.DataFrame:
    """Связь годовых изменений 2023→2024 с числом переключений в 2024 году.

    Используется только описательная ранговая корреляция. Пропуски исключаются
    для каждого показателя отдельно. Направление X_delta должно быть проверено
    по реализации шага 04; знаковая интерпретация без этого запрещена.
    """
    if not aligned.index.equals(delta.index):
        raise ValueError('Порядок муниципалитетов в X_delta не совпадает')
    months = list(aligned.columns)
    pairs = [(a,b) for a,b in zip(months[:-1], months[1:]) if pd.Period(b,'M').year == year]
    if not pairs:
        raise ValueError(f'Нет месячных переходов, заканчивающихся в {year}')
    switch_count = sum((aligned[a].to_numpy() != aligned[b].to_numpy()).astype(int)
                       for a,b in pairs)
    rows = []
    for col in delta.columns:
        name = str(col)
        if not name.endswith(f'__delta_{year-1}_{year}'):
            raise ValueError(f'Неожиданная структура X_delta: {col}')
        values = pd.to_numeric(delta[col], errors='raise').to_numpy(dtype=float, na_value=np.nan)
        valid = np.isfinite(values)
        if valid.sum() < 10 or np.std(switch_count[valid]) == 0:
            rho, rho_abs = np.nan, np.nan
        else:
            rho = float(spearmanr(values[valid], switch_count[valid]).statistic)
            rho_abs = float(spearmanr(np.abs(values[valid]), switch_count[valid]).statistic)
        rows.append({'indicator':name.split('__delta_')[0], 'n_valid':int(valid.sum()),
                     'n_missing':int((~valid).sum()),
                     'spearman_delta_vs_switches':rho,
                     'spearman_abs_delta_vs_switches':rho_abs,
                     'transition_pairs':len(pairs),
                     'mean_switches':float(switch_count[valid].mean()) if valid.any() else np.nan})
    return pd.DataFrame(rows)


def save_results(directory: str | Path, original_index: pd.Index, months: Sequence[str],
                 independent: pd.DataFrame, temporal: pd.DataFrame,
                 transitions_table: pd.DataFrame, stability_table: pd.DataFrame,
                 metadata: dict) -> None:
    """Только итоговые метки двух моделей, таблица переходов, метрики и JSON."""
    if not independent.index.equals(original_index) or not temporal.index.equals(original_index):
        raise ValueError('Индекс и порядок муниципалитетов изменился')
    if list(independent.columns) != list(months) or list(temporal.columns) != list(months):
        raise ValueError('Некорректный порядок месяцев')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    labels = pd.concat({'independent':independent, 'temporal':temporal}, axis=1)
    labels = labels.swaplevel(0,1,axis=1).sort_index(axis=1)
    labels.columns.names = ['month','model']
    labels.to_parquet(directory/'temporal_labels.parquet')
    transitions_table.to_parquet(directory/'transitions.parquet', index=False)
    stability_table.to_parquet(directory/'temporal_metrics.parquet', index=False)
    (directory/'temporal_experiment.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')


def plot_flow_three_periods(labels: pd.DataFrame, dates: Sequence[str], ax=None, min_flow_share: float = 0.01):
    """Трёхколоночная статическая alluvial-диаграмма без тысяч отдельных линий."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle
    if len(dates) != 3 or any(d not in labels.columns for d in dates):
        raise ValueError('Нужны три существующих месяца')
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))
    n = len(labels)
    palettes = plt.get_cmap('tab20')
    positions = []
    for x, month in enumerate(dates):
        counts = labels[month].value_counts().sort_index()
        start = 0
        pos = {}
        for j, (cluster, size) in enumerate(counts.items()):
            color = palettes(int(cluster)%20)
            ax.add_patch(Rectangle((x-0.055,start/n),.11,size/n,
                                   facecolor=color,edgecolor='white',linewidth=.7,alpha=.9))
            if size/n >= .05:
                ax.text(x, (start+size/2)/n, str(cluster),ha='center',va='center',fontsize=8)
            pos[cluster] = [start/n, (start+size)/n, color]
            start += size
        positions.append(pos)
    for t in range(2):
        tab = transition_matrix(labels[dates[t]],labels[dates[t+1]])
        out_pos = {a:v[0] for a,v in positions[t].items()}
        in_pos = {b:v[0] for b,v in positions[t+1].items()}
        for a in tab.index:
            for b in tab.columns:
                count = int(tab.loc[a,b])
                height = count/n
                if not count:
                    continue
                y0 = out_pos[a]; y1 = in_pos[b]
                if height >= min_flow_share:
                    poly = Polygon([(t+.06,y0), (t+.94,y1),
                                    (t+.94,y1+height), (t+.06,y0+height)],
                                   facecolor=positions[t][a][2],alpha=.23,edgecolor='none')
                    ax.add_patch(poly)
                out_pos[a] += height
                in_pos[b] += height
    ax.set(xlim=(-.2,2.2),ylim=(0,1),xticks=[0,1,2],xticklabels=list(dates),
           ylabel='Доля муниципалитетов',title='Переходы между экономическими типами')
    ax.grid(False)
    return ax


def _two_year_months(months: Sequence[str]) -> tuple[int, int, list[str]]:
    """Проверить две полные последовательные календарные годовые панели."""
    periods = pd.PeriodIndex(list(months), freq='M')
    if len(periods) != 24 or len(set(periods)) != 24:
        raise ValueError('Для календарного сравнения нужны ровно 24 различных месяца')
    years = sorted(set(periods.year))
    if len(years) != 2 or years[1] != years[0] + 1:
        raise ValueError('Нужны два последовательных календарных года')
    expected = [f'{y}-{m:02d}' for y in years for m in range(1, 13)]
    if list(months) != expected:
        raise ValueError('Нужны полные календарные годы по порядку январь–декабрь')
    return years[0], years[1], expected


def calendar_clr_repeatability(
    features: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Ретроспективное сравнение двух лет по центрированным CLR-отклонениям.

    Возвращает матрицу медианных расстояний (12×12), статистику для
    12 циклических календарных сдвигов и компактную диагностику по МО.
    Никакие исходные признаки, метки или индексы не изменяются.
    """
    y0, y1, months = _two_year_months(list(features))
    index = features[months[0]].index
    if not index.is_unique or index.hasnans or len(index) == 0:
        raise ValueError('Невалидные идентификаторы муниципалитетов')
    arrays = []
    p = features[months[0]].shape[1]
    first_columns = [str(c).removeprefix(f'{months[0]}__') for c in features[months[0]].columns]
    for m in months:
        x = features[m]
        categories = [str(c).removeprefix(f'{m}__') for c in x.columns]
        if not x.index.equals(index) or x.shape[1] != p or categories != first_columns:
            raise ValueError(f'Несопоставимые признаки или порядок ID в {m}')
        a = x.to_numpy(dtype=float)
        if not np.isfinite(a).all():
            raise ValueError(f'Недопустимые значения CLR в {m}')
        arrays.append(a)
    data = np.stack(arrays).reshape(2, 12, len(index), p)
    residual = data - data.mean(axis=1, keepdims=True)
    distances = np.empty((12, 12, len(index)), dtype=float)
    for a in range(12):
        for b in range(12):
            distances[a, b] = np.linalg.norm(residual[0, a] - residual[1, b], axis=1)
    month_names = [f'{m:02d}' for m in range(1, 13)]
    matrix = pd.DataFrame(np.median(distances, axis=2), index=month_names, columns=month_names)
    shift_rows = []
    for s in range(12):
        vals = distances[np.arange(12), (np.arange(12) + s) % 12]
        shift_rows.append({'calendar_shift': s, 'median_distance': float(np.median(vals)),
                           'mean_distance': float(vals.mean())})
    shifts = pd.DataFrame(shift_rows)
    matched = distances[np.arange(12), np.arange(12)].mean(axis=0)
    off_diagonal = distances[~np.eye(12, dtype=bool)].reshape(132, len(index)).mean(axis=0)
    gain = off_diagonal - matched
    summary = pd.DataFrame([{
        'year_from': y0, 'year_to': y1, 'municipalities': len(index),
        'matched_mean_distance': float(matched.mean()),
        'unmatched_mean_distance': float(off_diagonal.mean()),
        'median_municipal_advantage': float(np.median(gain)),
        'share_municipalities_positive_advantage': float(np.mean(gain > 0)),
        'matched_rank_among_12_shifts': int(shifts['mean_distance'].rank(method='min').iloc[0]),
    }])
    return matrix, shifts, summary


def calendar_cluster_repeatability(
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ARI разбиений между годами и повторяемость месячных переходов.

    ARI инвариантен к переименованию идентификаторов кластеров.
    Доля сменивших тип требует согласованных меток этапа 08/11.
    """
    y0, y1, months = _two_year_months(list(labels.columns))
    if not labels.index.is_unique or labels.index.hasnans or len(labels) == 0 or labels.isna().any().any():
        raise ValueError('Невалидные метки или идентификаторы муниципалитетов')
    arr = labels[months].to_numpy()
    if not np.isfinite(arr.astype(float)).all():
        raise ValueError('Метки должны быть конечными числовыми значениями')
    sim = np.empty((12, 12))
    for a in range(12):
        for b in range(12):
            sim[a, b] = adjusted_rand_score(arr[:, a], arr[:, 12 + b])
    names = [f'{m:02d}' for m in range(1, 13)]
    matrix = pd.DataFrame(sim, index=names, columns=names)
    shifts = pd.DataFrame([{
        'calendar_shift': s,
        'mean_ari': float(np.mean(sim[np.arange(12), (np.arange(12)+s) % 12])),
        'median_ari': float(np.median(sim[np.arange(12), (np.arange(12)+s) % 12])),
    } for s in range(12)])
    transition_rows = []
    for a in range(11):
        transition_rows.append({
            'month_pair': f'{a+1:02d}→{a+2:02d}',
            f'switch_share_{y0}': float(np.mean(arr[:, a] != arr[:, a+1])),
            f'switch_share_{y1}': float(np.mean(arr[:, 12+a] != arr[:, 13+a])),
        })
    return matrix, shifts, pd.DataFrame(transition_rows)
