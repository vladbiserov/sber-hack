"""Этап 07А: внешний годовой профиль разбиений и проверка атрибутивного слоя.

Не изменяет графы и метки этапов 06–07. Пропуски БДМО не заполняются.
Временная доступность годовых публикаций не предполагается известной.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kruskal, rankdata
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import RobustScaler

INDICATORS = (
    'wage', 'employment_per_1000', 'reporting_companies_per_1000',
    'profitable_share', 'investment_total_per_capita',
    'shop_floor_per_capita', 'population_density', 'lit_street_share',
)
# Единицы ниже - смысловые ориентиры по названиям и определению этапа 05.
# Для окончательной публикации сверить единицы со словарём этапа 04.
INDICATOR_INFO = {
    'wage': ('Оплата труда', 'денежных единиц/месяц (сверить с БДМО)'),
    'employment_per_1000': ('Занятые на 1000 жителей', 'человек/1000 жителей'),
    'reporting_companies_per_1000': ('Отчитывающиеся организации', 'организаций/1000 жителей'),
    'profitable_share': ('Доля прибыльных организаций', '%'),
    'investment_total_per_capita': ('Инвестиции на жителя', 'денежных единиц/житель (сверить с БДМО)'),
    'shop_floor_per_capita': ('Торговая площадь на жителя', 'м²/житель'),
    'population_density': ('Плотность населения', 'человек/км²'),
    'lit_street_share': ('Доля освещённых улиц', '%'),
}
POSITIVE_OR_ZERO = (
    'wage', 'employment_per_1000', 'reporting_companies_per_1000',
    'investment_total_per_capita', 'shop_floor_per_capita', 'population_density',
)
PERCENT = ('profitable_share', 'lit_street_share')


def validate_inputs(dynamic: pd.DataFrame, annual: pd.DataFrame,
                    labels: pd.DataFrame) -> None:
    """Строгое соответствие и порядок муниципалитетов; проверка меток/колонок."""
    if not dynamic.index.is_unique or dynamic.index.hasnans:
        raise ValueError('Неверный индекс муниципалитетов')
    for name, frame in [('X_annual', annual), ('labels', labels)]:
        if not frame.index.equals(dynamic.index):
            raise ValueError(f'{name}: индекс и порядок муниципалитетов должны совпадать с X_dynamic')
    expected = [f'{name}__{year}' for year in (2023, 2024) for name in INDICATORS]
    if list(annual.columns) != expected:
        raise ValueError('X_annual имеет неожиданную структуру или порядок столбцов')
    if not isinstance(labels.columns, pd.MultiIndex) or labels.columns.nlevels != 2:
        raise ValueError('labels.parquet должен иметь MultiIndex-колонки (month, method)')
    if labels.columns.has_duplicates or labels.isna().any().any():
        raise ValueError('Повторяющиеся колонки или пропущенные метки кластеров')
    if not all(pd.api.types.is_numeric_dtype(annual[c]) for c in annual):
        raise ValueError('Годовые показатели должны быть числовыми')
    if np.isinf(annual.to_numpy(dtype=float)).any():
        raise ValueError('Бесконечные значения годовых показателей')


def annual_view(annual: pd.DataFrame, year: int) -> pd.DataFrame:
    """Выбрать 8 показателей строго одного года в неизменном порядке."""
    cols = [f'{name}__{int(year)}' for name in INDICATORS]
    if any(col not in annual.columns for col in cols):
        raise KeyError(f'Нет всех восьми показателей для {year}')
    out = annual.loc[:, cols].copy()
    out.columns = list(INDICATORS)
    return out


def label_view(labels: pd.DataFrame, month: str, method: str) -> pd.Series:
    """Возвращает метки с исходным индексом; метки разных месяцев не совмещаются."""
    if (month, method) not in labels.columns:
        raise KeyError(f'Нет меток {month}, {method}')
    return labels[(month, method)].rename('cluster').copy()


def annual_profiles(annual: pd.DataFrame, labels: pd.DataFrame,
                    snapshots: list[tuple[str, int]], methods: tuple[str, ...]) -> pd.DataFrame:
    """Внешняя характеристика отдельных месячных разбиений без импутации."""
    rows = []
    for month, year in snapshots:
        features = annual_view(annual, year)
        for method in methods:
            partitions = label_view(labels, month, method)
            for cluster, idx in partitions.groupby(partitions, sort=True).groups.items():
                sub = features.loc[idx]
                for indicator in INDICATORS:
                    values = sub[indicator].dropna()
                    n = len(sub)
                    rows.append({
                        'month': month, 'year': year, 'method': method,
                        'cluster': int(cluster), 'indicator': indicator,
                        'cluster_size': n, 'observed': int(len(values)),
                        'missing_fraction': 1 - len(values) / n,
                        'median': float(values.median()) if len(values) else np.nan,
                        'q25': float(values.quantile(.25)) if len(values) else np.nan,
                        'q75': float(values.quantile(.75)) if len(values) else np.nan,
                    })
    return pd.DataFrame(rows)


def rank_separation(annual: pd.DataFrame, labels: pd.DataFrame,
                    snapshots: list[tuple[str, int]], methods: tuple[str, ...]) -> pd.DataFrame:
    """Описательная η² по рангам; Kruskal p только при >=5 наблюдениях в каждой группе.

    Сравнение разных методов по η² - описательное, т.к. число кластеров различается.
    """
    rows = []
    for month, year in snapshots:
        annual_year = annual_view(annual, year)
        for method in methods:
            lab = label_view(labels, month, method)
            for indicator in INDICATORS:
                usable = annual_year[indicator].notna()
                obs = annual_year.loc[usable, indicator].to_numpy(float)
                groups = lab.loc[usable].to_numpy()
                unique, inv, sizes = np.unique(groups, return_inverse=True, return_counts=True)
                eta = np.nan
                p_value = np.nan
                if len(unique) > 1 and len(obs) > len(unique):
                    ranks = rankdata(obs, method='average')
                    total_ss = float(np.square(ranks-ranks.mean()).sum())
                    if total_ss > 0:
                        means = np.bincount(inv, weights=ranks) / sizes
                        between_ss = float((sizes * np.square(means - ranks.mean())).sum())
                        eta = float(np.clip(between_ss / total_ss, 0, 1))
                    if sizes.min() >= 5:
                        p_value = float(kruskal(*(obs[groups == k] for k in unique)).pvalue)
                rows.append({'month': month, 'year': year, 'method': method,
                             'indicator': indicator, 'n_observed': len(obs),
                             'clusters': len(unique), 'min_observed_cluster': int(sizes.min()),
                             'eta2_ranks': eta, 'kruskal_p': p_value})
    table = pd.DataFrame(rows)
    # Поправка Бенджамини–Хохберга в рамках (месяц, метод): 8 гипотез.
    table['kruskal_q_bh'] = np.nan
    for (_, _), subset in table.groupby(['month', 'method']):
        valid = subset.kruskal_p.dropna().sort_values()
        if len(valid):
            rank = np.arange(1, len(valid)+1, dtype=float)
            corrected = np.minimum.accumulate((valid.to_numpy()*len(valid)/rank)[::-1])[::-1]
            table.loc[valid.index, 'kruskal_q_bh'] = np.clip(corrected, 0, 1)
    return table


def prepare_lagged_annual(annual: pd.DataFrame, month: str, year: int,
                          minimum_n: int = 500) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Лаговый год (строго меньше года целевого месяца); без заполнения пропусков.

    Дата публикации не проверяется: результат можно использовать только как
    ретроспективную чувствительность, не как доказательство доступности as-of.
    """
    m = pd.Period(month, freq='M')
    if int(year) >= m.year:
        raise ValueError('Запрещены показатели текущего или будущего отчётного года')
    x = annual_view(annual, year)
    mask = x.notna().all(axis=1)
    if int(mask.sum()) < minimum_n:
        raise ValueError(f'Недостаточно полных наблюдений: {mask.sum()} < {minimum_n}')
    chosen = x.loc[mask].copy()
    for c in POSITIVE_OR_ZERO:
        if (chosen[c] < 0).any():
            raise ValueError(f'{c}: отрицательные значения не допускаются для log1p')
    for c in PERCENT:
        if ((chosen[c] < 0) | (chosen[c] > 100)).any():
            raise ValueError(f'{c}: значение за пределами [0, 100]')
    z = chosen.copy()
    z.loc[:, list(POSITIVE_OR_ZERO)] = np.log1p(z.loc[:, list(POSITIVE_OR_ZERO)])
    z.loc[:, list(PERCENT)] = z.loc[:, list(PERCENT)] / 100
    z_scaled = pd.DataFrame(RobustScaler().fit_transform(z), index=z.index, columns=z.columns)
    return z_scaled, mask.rename('annual_complete'), x


def balanced_blocks(monthly: pd.DataFrame, annual: pd.DataFrame,
                    alpha: float) -> pd.DataFrame:
    """Заданный вклад α в полную межмуниципальную дисперсию двух блоков.

    Нормируем каждый центрированный блок на sqrt(sum_k Var(col_k)), затем
    масштабируем sqrt(alpha) и sqrt(1-alpha). Индивидуальные CLR-оси
    не стандартизируем, чтобы сохранить их евклидову геометрию.
    """
    if not (0 <= alpha <= 1):
        raise ValueError('alpha должен принадлежать [0,1]')
    if not monthly.index.equals(annual.index):
        raise ValueError('Блоки имеют разные индексы муниципалитетов')
    arrays = []
    for data, weight, prefix in [(monthly, alpha, 'clr__'), (annual, 1-alpha, 'annual__')]:
        v = data.to_numpy(dtype=float)
        if not np.isfinite(v).all():
            raise ValueError('Пропуски и бесконечные значения не допускаются')
        v = v - v.mean(axis=0, keepdims=True)
        ss = float(v.var(axis=0, ddof=0).sum())
        if ss <= 0:
            raise ValueError(f'У {prefix} нет межмуниципальной вариации')
        norm = np.sqrt(weight) * v / np.sqrt(ss)
        arrays.append(pd.DataFrame(norm, index=data.index,
                                   columns=[prefix+str(c) for c in data.columns]))
    return pd.concat(arrays, axis=1)


def attribution_experiment(monthly_clr: pd.DataFrame, annual: pd.DataFrame,
                           month: str, annual_year: int, k: int,
                           alphas: list[float], seeds: list[int],
                           sample_size: int = 500, minimum_n: int = 500,
                           original_labels: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Сравнение K-Means на одинаковом complete-case наборе и числе кластеров.

    Годовой слой - лаговый; дата публикации не установлена. Нельзя считать
    результаты временно-валидированной моделью без дополнительной проверки.
    """
    if 1.0 not in alphas or len(set(alphas)) != len(alphas):
        raise ValueError('В alphas должен быть единственный контроль alpha=1')
    if len(seeds) < 2 or not 2 <= k < len(monthly_clr):
        raise ValueError('Нужно >=2 seeds и допустимое число кластеров')
    a, mask, _ = prepare_lagged_annual(annual, month, annual_year, minimum_n)
    m = monthly_clr.loc[mask]
    if not m.index.equals(a.index):
        raise ValueError('Нарушен порядок индексов после complete-case')
    baseline = {}
    rows = []
    results = {}
    rng = np.random.default_rng(2026)
    subset = np.sort(rng.choice(len(m), size=min(sample_size, len(m)), replace=False))
    for alpha in sorted(alphas, reverse=True):
        Z = balanced_blocks(m, a, float(alpha))
        for seed in seeds:
            lab = KMeans(n_clusters=k, random_state=int(seed), n_init=10).fit_predict(Z)
            results[(alpha, seed)] = lab
            if alpha == 1.0:
                baseline[seed] = lab
            sub_labels = lab[subset]
            sil = (silhouette_score(m.to_numpy()[subset], sub_labels)
                   if 1 < len(np.unique(sub_labels)) < len(subset) else np.nan)
            ch = (calinski_harabasz_score(m.to_numpy(), lab)
                  if 1 < len(np.unique(lab)) < len(lab) else np.nan)
            sil_z = (silhouette_score(Z.to_numpy()[subset], sub_labels)
                     if 1 < len(np.unique(sub_labels)) < len(subset) else np.nan)
            rows.append({'month': month, 'annual_year': int(annual_year),
                         'alpha': float(alpha), 'seed': int(seed), 'n': len(m),
                         'n_excluded': int((~mask).sum()), 'clusters': len(np.unique(lab)),
                         'min_cluster': int(np.unique(lab, return_counts=True)[1].min()),
                         'largest_share': float(np.unique(lab, return_counts=True)[1].max()/len(lab)),
                         'silhouette_clr': float(sil),
                         'calinski_harabasz_clr': float(ch),
                         'silhouette_own_space': float(sil_z),
                         'ari_vs_07': float(adjusted_rand_score(original_labels.loc[mask].to_numpy(), lab))
                              if original_labels is not None else np.nan})
    table = pd.DataFrame(rows)
    for alpha in alphas:
        matches = [adjusted_rand_score(results[(alpha, s)], baseline[s]) for s in seeds]
        repeats = [adjusted_rand_score(results[(alpha, s)], results[(alpha, t)])
                   for s, t in combinations(seeds, 2)]
        table.loc[table.alpha == alpha, 'ari_vs_clr_baseline'] = float(np.mean(matches))
        table.loc[table.alpha == alpha, 'ari_seeds'] = float(np.mean(repeats))
    return table, mask
