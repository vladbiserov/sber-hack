"""Этап 11: воспроизводимое сравнение устойчивости уже отобранных моделей.

Не выполняет подбор параметров заново и не изменяет результаты 05–10.
Повторные кластеризации используются только как ограниченные stress-тесты.
"""
from __future__ import annotations

import json
from itertools import combinations

from scipy.optimize import linear_sum_assignment
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .temporal_analysis import time_metrics, attribute_change_association, run_lambda_grid
from .cluster_models import weighted_modularity, run_partition
from .annual_attribution import annual_view, rank_separation


def validate_candidate(reference: pd.DataFrame, candidate: pd.DataFrame) -> None:
    """Проверить совпадение муниципалитетов, месяцев и корректность меток."""
    if not reference.index.is_unique or reference.index.hasnans:
        raise ValueError('Неуникальные/отсутствующие ID муниципалитетов')
    if not reference.index.equals(candidate.index) or not reference.columns.equals(candidate.columns):
        raise ValueError('Различаются ID, порядок муниципалитетов или месяцы')
    if reference.empty or reference.isna().any().any() or candidate.isna().any().any():
        raise ValueError('Пустые либо неполные метки кластеров')


def compare_partitions(reference: pd.DataFrame, alternatives: Mapping[str, pd.DataFrame],
                       months: Sequence[str] | None = None) -> pd.DataFrame:
    """ARI/NMI одинаковых объектов и месяцев; перестановка меток не важна."""
    selected_months = list(reference.columns) if months is None else list(months)
    if not selected_months or any(m not in reference.columns for m in selected_months):
        raise ValueError('Неизвестные месяцы для сравнения')
    rows = []
    for name, alternative in alternatives.items():
        validate_candidate(reference, alternative)
        for month in selected_months:
            a = reference[month].to_numpy()
            b = alternative[month].to_numpy()
            sizes = np.unique(b, return_counts=True)[1]
            rows.append({'configuration': str(name), 'month': month,
                         'ari_to_reference': float(adjusted_rand_score(a, b)),
                         'nmi_to_reference': float(normalized_mutual_info_score(a, b)),
                         'n_clusters': int(len(sizes)),
                         'largest_share': float(sizes.max()/len(b))})
    return pd.DataFrame(rows)


def align_to_reference(reference: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    """Сопоставить номера типов с эталоном по каждому месяцу (Hungarian)."""
    validate_candidate(reference, candidate)
    expected = set(reference.iloc[:, 0].unique())
    result = pd.DataFrame(index=reference.index)
    for month in reference.columns:
        ref = reference[month].to_numpy()
        current = candidate[month].to_numpy()
        ref_ids = np.unique(ref)
        candidate_ids = np.unique(current)
        if set(ref_ids) != expected or len(candidate_ids) != len(ref_ids):
            raise ValueError(f'{month}: число или идентификаторы кластеров различаются')
        overlap = np.array([
            [np.count_nonzero((ref == a) & (current == b)) for b in candidate_ids]
            for a in ref_ids
        ])
        rows, cols = linear_sum_assignment(-overlap)
        mapping = dict(zip(candidate_ids[cols], ref_ids[rows]))
        result[month] = candidate[month].map(mapping)
    validate_candidate(reference, result)
    return result


def temporal_kmeans_repeat(reference: pd.DataFrame,
                           features: Mapping[str, pd.DataFrame],
                           graphs: Mapping[str, object], n_clusters: int,
                           seed: int, lambda_value: float,
                           graph_beta: float) -> pd.DataFrame:
    """Повторить K-Means и временную регуляризацию для проверки устойчивости."""
    if list(features) != list(reference.columns) or list(graphs) != list(reference.columns):
        raise ValueError('Несовпадение месяцев в признаках, графах и разбиении')
    for month in reference.columns:
        if not features[month].index.equals(reference.index):
            raise ValueError(f'{month}: неверный порядок муниципалитетов')
    raw = pd.DataFrame({
        month: run_partition('kmeans', features[month], graphs[month],
                             int(n_clusters), int(seed))
        for month in reference.columns
    }, index=reference.index)
    aligned = align_to_reference(reference, raw)
    lam = float(lambda_value)
    grid = [0.0] if lam == 0.0 else [0.0, lam]
    _, paths, _ = run_lambda_grid(aligned, features, graphs, grid,
                                  graph_beta=float(graph_beta))
    return paths[lam]


def seed_repeatability(paths: Mapping[int, pd.DataFrame]) -> pd.DataFrame:
    """Попарный ARI/NMI между seed по каждому месяцу, без переобучения."""
    if len(paths) < 2:
        raise ValueError('Нужны минимум два случайных состояния')
    seeds = sorted(paths)
    first = paths[seeds[0]]
    for seed in seeds[1:]:
        validate_candidate(first, paths[seed])
    rows = []
    for a, b in combinations(seeds, 2):
        for month in first.columns:
            rows.append({'seed_a': int(a), 'seed_b': int(b), 'month': str(month),
                         'ari': float(adjusted_rand_score(paths[a][month], paths[b][month])),
                         'nmi': float(normalized_mutual_info_score(paths[a][month], paths[b][month]))})
    return pd.DataFrame(rows)


def model_diagnostics(paths: Mapping[str, pd.DataFrame],
                      features: Mapping[str, pd.DataFrame], graphs: Mapping[str, object],
                      graph_by_candidate: Mapping[str, Mapping[str, object]] | None = None) -> pd.DataFrame:
    """Временная устойчивость, согласованность с изменениями CLR и модульность.

    Для всех кандидатов проверяем изменения на одном эталонном CLR-пространстве.
    Если кандидат использует альтернативные графы, модульность вычисляется на них
    и это явно помечается, чтобы не сравнивать её напрямую с основным графом.
    """
    base_index = next(iter(features.values())).index
    months = list(features)
    rows = []
    for name, path in paths.items():
        if not path.index.equals(base_index) or list(path.columns) != months:
            raise ValueError(f'Неверное выравнивание модели {name}')
        stability = time_metrics(path)
        assoc = attribute_change_association(path, features)
        graph_set = graph_by_candidate.get(name, graphs) if graph_by_candidate else graphs
        mod = [weighted_modularity(graph_set[m], path[m].to_numpy()) for m in months]
        cluster_sizes = [path[m].value_counts().max()/len(path) for m in months]
        rows.append({
            'configuration': name, 'months': len(months),
            'ari_adjacent_median': float(stability.ari.median()),
            'nmi_adjacent_median': float(stability.nmi.median()),
            'switch_share_median': float(stability.switch_share.median()),
            'modularity_median': float(np.nanmedian(mod)),
            'modularity_graph': 'candidate' if graph_set is not graphs else 'baseline',
            'max_cluster_share_median': float(np.median(cluster_sizes)),
            'median_clr_change_switch': float(assoc.median_clr_change_switch.dropna().median()),
            'median_clr_change_stay': float(assoc.median_clr_change_stay.dropna().median()),
            'change_switch_spearman_median': float(assoc.spearman_change_vs_switch.dropna().median()),
        })
    return pd.DataFrame(rows)


def annual_expost_separation(annual: pd.DataFrame, partitions: Mapping[str, pd.DataFrame],
                             month: str = '2024-12', year: int = 2024) -> pd.DataFrame:
    """Ранговая η² годовых показателей как ВНЕШНИЙ ретроспективный критерий.

    Для каждого индикатора используется собственная наблюдаемая подвыборка;
    метрика не подтверждает доступность БДМО на указанную дату.
    """
    # Используем проверенную реализацию этапа 07А, которая не заполняет NaN.
    if not partitions:
        raise ValueError('Нет разбиений')
    annual_view(annual, year)  # валидировать наличие всех восьми показателей
    first = next(iter(partitions.values()))
    if not first.index.equals(annual.index):
        raise ValueError('Годовые показатели и разбиения имеют разные ID')
    columns = {}
    for name, frame in partitions.items():
        if not frame.index.equals(annual.index) or month not in frame.columns:
            raise ValueError(f'Неверный индекс или отсутствует месяц {month}: {name}')
        columns[(month, name)] = frame[month]
    labels = pd.DataFrame(columns, index=annual.index)
    labels.columns = pd.MultiIndex.from_tuples(labels.columns, names=['month','method'])
    sep = rank_separation(annual, labels, [(month, year)], tuple(partitions))
    return sep.groupby('method', as_index=False).agg(
        eta2_rank_median=('eta2_ranks', 'median'),
        valid_indicators=('eta2_ranks', 'count'),
        min_n_observed=('n_observed', 'min'),
    ).rename(columns={'method':'configuration'})


def final_metadata(config: dict, chosen_id: str, index: pd.Index,
                   months: Sequence[str], diagnostics: pd.DataFrame) -> dict:
    """Минимальная проверяемая метаинформация для передачи этапу 12."""
    if chosen_id not in set(diagnostics['configuration']):
        raise ValueError('Выбранная модель отсутствует в сравнении')
    if not index.is_unique or index.hasnans or len(set(months)) != len(months):
        raise ValueError('Недопустимые ID или месяцы')
    is_temporal = chosen_id.startswith('temporal_')
    lambda_selected = float(chosen_id.rsplit('_lambda_', 1)[1]) if is_temporal else None
    representation = 'raw' if '_raw_' in chosen_id else 'clr'
    return {'selected_configuration': chosen_id,
            'n_municipalities': int(len(index)), 'months': list(months),
            'graph_configuration': config.get('graph_selection_06'),
            'stage07_parameters': config['selected_parameters'],
            'temporal_settings': {'lambda': lambda_selected,
                                  'graph_beta': config['temporal_08']['graph_beta'] if is_temporal else None},
            'feature_space': f'monthly_{representation}_6',
            'economic_interpretation': 'stage10_retro_expost',
            'source_config': 'configs/clustering_experiment.json',
            'note': 'Результат этапа 11 после проверки outputs; не прогнозная as-of модель'}


def save_final(directory: str | Path, index: pd.Index, months: Sequence[str],
               final_labels: pd.DataFrame, comparison: pd.DataFrame,
               metadata: dict) -> tuple[Path, Path, Path]:
    """Записать только согласованные метки, краткую сводку и JSON.

    Вызывается вручную после научного выбора; не сохраняет промежуточные разбиения.
    """
    if not final_labels.index.equals(index) or list(final_labels.columns) != list(months):
        raise ValueError('Неверные ID или порядок месяцев в итоговых метках')
    if final_labels.isna().any().any() or comparison.empty:
        raise ValueError('Пропущенные метки или пустая таблица сравнения')
    if metadata.get('selected_configuration') is None:
        raise ValueError('Не утверждена окончательная конфигурация')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    labels_path = directory / 'final_labels.parquet'
    report_path = directory / 'robustness_comparison.parquet'
    config_path = directory / 'final_model.json'
    final_labels.to_parquet(labels_path)
    comparison.to_parquet(report_path, index=False)
    config_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    return labels_path, report_path, config_path


def graph_ablation_paths(reference: pd.DataFrame,
                         features: Mapping[str, pd.DataFrame],
                         graphs: Mapping[str, object],
                         betas: Sequence[float], lambda_value: float
                         ) -> tuple[dict[float, pd.DataFrame], pd.DataFrame]:
    """Controlled graph ablation: same monthly prototypes, graph and lambda, varied beta.

    Calling run_lambda_grid for each beta does not refit K-Means. beta=0
    switches off ONLY the within-month graph penalty, not temporal smoothing.
    """
    values = [float(b) for b in betas]
    if not values or len(values) != len(set(values)) or 0.0 not in values:
        raise ValueError('Нужен контроль beta=0 и уникальные коэффициенты')
    if not all(np.isfinite(values)) or any(b < 0 for b in values):
        raise ValueError('beta должен быть конечным неотрицательным числом')
    lam = float(lambda_value)
    if not np.isfinite(lam) or lam < 0:
        raise ValueError('Некорректный lambda')
    if list(reference.columns) != list(features) or list(reference.columns) != list(graphs):
        raise ValueError('Месяцы разбиения, признаков и графов различаются')
    for m in reference.columns:
        if not reference.index.equals(features[m].index):
            raise ValueError(f'Порядок муниципалитетов отличается в {m}')
    grid = [0.] if lam == 0 else [0., lam]
    paths, rows = {}, []
    for beta in values:
        report, trials, _ = run_lambda_grid(reference, features, graphs, grid, graph_beta=beta)
        paths[beta] = trials[lam]
        rows.append(report.loc[report['lambda'].eq(lam)].iloc[0].to_dict())
    return paths, pd.DataFrame(rows).sort_values('graph_beta').reset_index(drop=True)


def graph_ablation_monthly(reference: pd.DataFrame, paths: Mapping[float, pd.DataFrame],
                            graphs: Mapping[str, object]) -> pd.DataFrame:
    """Month-specific graph contribution measured relative to beta=0.

    Identical state IDs are inherited from reference, so the changed fraction
    is meaningful without an extra label alignment. ARI/NMI are permutation-invariant.
    Modularity is calculated for ALL candidates against identical graphs.
    """
    if 0.0 not in paths:
        raise ValueError('Отсутствует контроль beta=0')
    no_graph = paths[0.0]
    validate_candidate(reference, no_graph)
    rows = []
    for beta, labels in sorted(paths.items()):
        validate_candidate(no_graph, labels)
        for m in no_graph.columns:
            base = no_graph[m].to_numpy()
            current = labels[m].to_numpy()
            counts = np.unique(current, return_counts=True)[1]
            rows.append({
                'month': str(m), 'beta': float(beta),
                'ari_to_beta0': float(adjusted_rand_score(base, current)),
                'nmi_to_beta0': float(normalized_mutual_info_score(base, current)),
                'changed_share_to_beta0': float(np.mean(base != current)),
                'changed_count_to_beta0': int(np.count_nonzero(base != current)),
                'clusters': int(len(counts)),
                'largest_cluster_share': float(counts.max() / len(current)),
                'modularity_same_graph': float(weighted_modularity(graphs[m], current)),
            })
    return pd.DataFrame(rows)


def graph_ablation_changes(paths: Mapping[float, pd.DataFrame],
                           features: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Test whether transitions are accompanied by changes in monthly CLR values.

    All candidate models are evaluated in the exact same CLR space. No observed
    association is interpreted as a causal effect or proof of a real shock.
    """
    rows = []
    for beta, labels in sorted(paths.items()):
        if list(labels.columns) != list(features):
            raise ValueError('Не совпадают месяцы меток и признаков')
        for previous, current in zip(labels.columns[:-1], labels.columns[1:]):
            if not labels.index.equals(features[previous].index) or not labels.index.equals(features[current].index):
                raise ValueError('Несовпадение порядка муниципалитетов')
            magnitude = np.linalg.norm(
                features[current].to_numpy(float)-features[previous].to_numpy(float), axis=1)
            switched = labels[previous].to_numpy() != labels[current].to_numpy()
            stay = ~switched
            rows.append({
                'beta': float(beta), 'month_to': str(current),
                'transition_share': float(switched.mean()),
                'n_switch': int(switched.sum()),
                'clr_change_switch': float(np.median(magnitude[switched])) if switched.any() else np.nan,
                'clr_change_stay': float(np.median(magnitude[stay])) if stay.any() else np.nan,
                'share_change_top_quartile_switch': float(np.mean(magnitude[switched] >= np.quantile(magnitude, .75)))
                                                    if switched.any() else np.nan,
            })
    return pd.DataFrame(rows)


def unstable_seed_diagnostics(paths: Mapping[int, pd.DataFrame],
                              reference: pd.DataFrame,
                              features: Mapping[str, pd.DataFrame]
                              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find weak seed-month pairs and diagnose CLR centroid boundary margins.

    Paths must have globally matched numeric type IDs, as created by
    temporal_kmeans_repeat; the code never relabels on arbitrary numeric order.
    """
    if len(paths) < 2:
        raise ValueError('Нужны минимум два запуска')
    for labels in paths.values():
        validate_candidate(reference, labels)
    if list(features) != list(reference.columns):
        raise ValueError('Не совпадают месяцы признаков')
    scores = seed_repeatability(paths)
    per_month = scores.groupby('month').agg(
        worst_pair_ari=('ari', 'min'), median_pair_ari=('ari', 'median'),
        worst_pair_nmi=('nmi', 'min')).reset_index()
    seed_arrays = [paths[k] for k in sorted(paths)]
    cluster_rows = []
    for m in reference.columns:
        ref = reference[m].to_numpy()
        X = features[m].to_numpy(dtype=float)
        if not features[m].index.equals(reference.index):
            raise ValueError(f'Неверный индекс в {m}')
        cluster_ids = np.unique(ref)
        centers = np.vstack([X[ref == c].mean(axis=0) for c in cluster_ids])
        dist = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        first_two = np.partition(dist, kth=1, axis=1)[:, :2]
        margin = (first_two[:, 1] - first_two[:, 0]) / (first_two[:, 1] + 1e-12)
        predictions = np.stack([p[m].to_numpy() for p in seed_arrays])
        disagrees = np.any(predictions != predictions[0:1], axis=0)
        for c in cluster_ids:
            group = ref == c
            uncertain = group & disagrees
            agreed = group & ~disagrees
            cluster_rows.append({
                'month': str(m), 'cluster': int(c), 'size': int(group.sum()),
                'disagreement_share': float(disagrees[group].mean()),
                'median_margin_disagree': float(np.median(margin[uncertain])) if uncertain.any() else np.nan,
                'median_margin_agree': float(np.median(margin[agreed])) if agreed.any() else np.nan,
            })
        per_month.loc[per_month.month.eq(m), 'disagreement_share'] = float(disagrees.mean())
        per_month.loc[per_month.month.eq(m), 'median_margin_disagree'] = (
            float(np.median(margin[disagrees])) if disagrees.any() else np.nan)
        per_month.loc[per_month.month.eq(m), 'median_margin_agree'] = (
            float(np.median(margin[~disagrees])) if (~disagrees).any() else np.nan)
    return per_month, pd.DataFrame(cluster_rows)
