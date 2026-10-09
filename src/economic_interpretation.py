"""Этап 10: описательная экономическая типология сохранённых временных кластеров.

Результаты кластеризации не пересчитываются. Годовые признаки используются только
для последующей экономической интерпретации, пропуски не импутируются.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .annual_attribution import INDICATORS, INDICATOR_INFO, annual_view
from .feature_analysis import SHARES, monthly_view, transform_shares


def validate_sources(dynamic: pd.DataFrame, annual: pd.DataFrame, delta: pd.DataFrame,
                     aligned: pd.DataFrame, months: Sequence[str]) -> None:
    """Проверить строгие идентификаторы, размерности и контракт сохранённых меток."""
    idx = dynamic.index
    if not idx.is_unique or idx.hasnans:
        raise ValueError('Некорректные идентификаторы муниципалитетов')
    for name, df in [('annual', annual), ('delta', delta), ('labels', aligned)]:
        if not df.index.equals(idx):
            raise ValueError(f'{name}: несовпадение индексов или порядка муниципалитетов')
    if list(aligned.columns) != list(months) or len(months) != 24:
        raise ValueError('Ожидались 24 месяца в правильном порядке')
    if aligned.isna().any().any() or not all(aligned[c].nunique() >= 2 for c in months):
        raise ValueError('Недопустимое разбиение или пропущенные метки')
    annual_expected = [f'{k}__{year}' for year in (2023, 2024) for k in INDICATORS]
    delta_expected = [f'{k}__delta_2023_2024' for k in INDICATORS]
    if list(annual.columns) != annual_expected or list(delta.columns) != delta_expected:
        raise ValueError('Неожиданные столбцы X_annual или X_delta')
    for name, df in [('annual', annual), ('delta', delta)]:
        if not all(pd.api.types.is_numeric_dtype(df[c]) for c in df):
            raise ValueError(f'{name}: все признаки должны быть числовыми')
        if np.isinf(df.to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError(f'{name}: обнаружены бесконечные значения')


def validate_delta_identity(annual: pd.DataFrame, delta: pd.DataFrame,
                            atol: float = 1e-6) -> pd.DataFrame:
    """Проверить, что X_delta — действительно разность 2024−2023, включая NA."""
    rows = []
    for k in INDICATORS:
        a, b = annual[f'{k}__2023'], annual[f'{k}__2024']
        d = delta[f'{k}__delta_2023_2024']
        expected = b - a
        present = expected.notna() & d.notna()
        matches_missing = expected.isna().equals(d.isna())
        matches_numbers = np.allclose(expected[present].to_numpy(dtype=float),
                                      d[present].to_numpy(dtype=float),
                                      rtol=1e-10, atol=atol)
        rows.append({'indicator':k, 'n_checked':int(present.sum()),
                     'missingness_equal':bool(matches_missing),
                     'values_equal':bool(matches_numbers)})
    out = pd.DataFrame(rows)
    if not (out['missingness_equal'].all() and out['values_equal'].all()):
        raise ValueError('X_delta не соответствует простым разностям 2024−2023; проверьте функцию flatten_annual')
    return out


def profile_panel(dynamic: pd.DataFrame, mapping: Mapping, annual: pd.DataFrame,
                  aligned: pd.DataFrame, snapshots: Sequence[tuple[str, int]]) -> pd.DataFrame:
    """Потребительские средние/медианы и годовые распределения; NA исключаются по признаку.

    Одни и те же ID, но составы кластеров могут изменяться между снимками.
    """
    rows = []
    for month, year in snapshots:
        month = str(month)
        if year != pd.Period(month, freq='M').year:
            raise ValueError('Год контекста и год месяца должны совпадать')
        shares = monthly_view(dynamic, mapping, month)
        context = annual_view(annual, year)
        labels = aligned[month]
        for family, panel in [('consumer_share', shares), ('annual_bdmo', context)]:
            for indicator in panel.columns:
                all_values = panel[indicator].dropna()
                baseline_mean = float(all_values.mean()) if len(all_values) else np.nan
                baseline_median = float(all_values.median()) if len(all_values) else np.nan
                baseline_iqr = float(all_values.quantile(.75)-all_values.quantile(.25)) if len(all_values) else np.nan
                for cluster, ids in labels.groupby(labels, sort=True).groups.items():
                    obs = panel.loc[ids, indicator].dropna().astype(float)
                    n = len(ids)
                    med = float(obs.median()) if len(obs) else np.nan
                    avg = float(obs.mean()) if len(obs) else np.nan
                    if family == 'consumer_share':
                        contrast = (avg - baseline_mean) * 100  # процентные пункты
                        unit = 'percentage_points_mean'
                    else:
                        contrast = ((med-baseline_median)/baseline_iqr if baseline_iqr > 0 else np.nan)
                        unit = 'median_difference_over_global_IQR'
                    rows.append({'month':month,'year':year,'cluster':int(cluster),
                        'family':family,'indicator':indicator,'cluster_size':int(n),
                        'observed':int(len(obs)),'coverage':len(obs)/n,
                        'mean':avg,'median':med,
                        'q25':float(obs.quantile(.25)) if len(obs) else np.nan,
                        'q75':float(obs.quantile(.75)) if len(obs) else np.nan,
                        'baseline_mean':baseline_mean,'baseline_median':baseline_median,
                        'baseline_iqr':baseline_iqr,'contrast':contrast,'contrast_unit':unit})
    return pd.DataFrame(rows)


def cluster_distinctions(profiles: pd.DataFrame, month: str,
                         min_observed: int = 20, min_coverage: float = 0.5,
                         top_n: int = 3) -> pd.DataFrame:
    """Отличительные признаки, отдельно для долей (п.п.) и БДМО (ед. IQR)."""
    x = profiles.loc[(profiles.month == month) &
                     (profiles.observed >= min_observed) &
                     (profiles.coverage >= min_coverage) &
                     profiles.contrast.notna()].copy()
    x['abs_contrast'] = x.contrast.abs()
    x = x.sort_values(['cluster','family','abs_contrast'],ascending=[True,True,False])
    return x.groupby(['cluster','family'],sort=True).head(top_n)[
        ['cluster','family','indicator','contrast','contrast_unit','coverage','observed']].reset_index(drop=True)


def interpretation_candidates(profiles: pd.DataFrame, month: str) -> pd.DataFrame:
    """Структурированная основа типологии, без автоматически выдуманных названий."""
    important = cluster_distinctions(profiles, month)
    out = []
    size = profiles[(profiles.month == month) & (profiles.family == 'consumer_share') &
                    (profiles.indicator == SHARES[0])].set_index('cluster')['cluster_size']
    for cid in sorted(size.index):
        chosen = important[important.cluster == cid]
        consumer = chosen[chosen.family == 'consumer_share']
        bdmo = chosen[chosen.family == 'annual_bdmo']
        fmt = lambda df: '; '.join(f'{r.indicator}: {r.contrast:+.2f}' for r in df.itertuples())
        out.append({'cluster':int(cid), 'month':month,'n_municipalities':int(size.loc[cid]),
                    'provisional_name':f'Тип {cid} — требует экономической интерпретации',
                    'consumer_contrasts_pp':fmt(consumer),
                    'annual_contrasts_iqr':fmt(bdmo),
                    'interpretation_status':'pending_review'})
    return pd.DataFrame(out)


_ID_NAMES = ('territory_id','municipality_id','mo_id','oktmo','id','territory_code')
_NAME_NAMES = ('municipality_name','territory_name','name_mo','mo_name','name','municipality','наименование мо','муниципальное образование')
_REGION_NAMES = ('region_name','subject_name','federal_subject','region','subject','субъект рф','регион')


def read_municipality_lookup(path: str | Path, ids: pd.Index,
                             id_col: str | None = None, name_col: str | None = None,
                             region_col: str | None = None) -> pd.DataFrame:
    """Строго проверить местный справочник; без географических догадок и заполнения имен."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Нет справочника: {path}')
    if path.suffix.lower() == '.parquet':
        lookup = pd.read_parquet(path)
    elif path.suffix.lower() in ('.csv', '.tsv'):
        lookup = pd.read_csv(path, sep='\t' if path.suffix.lower()=='.tsv' else None,
                             engine='python', dtype=str)
    elif path.suffix.lower() in ('.xlsx', '.xls'):
        lookup = pd.read_excel(path, dtype=str)
    else:
        raise ValueError(f'Не поддерживается формат справочника: {path.suffix}')
    cols = list(lookup.columns)
    def choose(preferred, aliases, role):
        if preferred is not None:
            if preferred not in cols:
                raise KeyError(f'{role}: столбец {preferred!r} отсутствует. Доступны: {cols}')
            return preferred
        matches = [col for col in cols if str(col).casefold().strip() in aliases]
        if len(matches) != 1:
            raise ValueError(f'{role}: не удалось однозначно определить колонку ({matches}); доступны {cols}. Укажите её в конфиге.')
        return matches[0]
    ic = choose(id_col, _ID_NAMES, 'ID')
    nc = choose(name_col, _NAME_NAMES, 'название муниципалитета')
    rc = choose(region_col, _REGION_NAMES, 'регион')
    ref = lookup[[ic,nc,rc]].copy()
    ref.columns = ['municipality_id','municipality_name','region']
    for col in ref:
        ref[col] = ref[col].astype('string').str.strip()
        ref[col] = ref[col].replace('', pd.NA)
    ref = ref.dropna(subset=['municipality_id'])
    # Разрешены только идентичные повторные записи, иначе выбор года неоднозначен.
    duplicates = ref.groupby('municipality_id')[['municipality_name','region']].nunique(dropna=False)
    if (duplicates > 1).any(axis=None):
        raise ValueError('В справочнике противоречивые названия или регионы для одного ID')
    ref = ref.drop_duplicates('municipality_id').set_index('municipality_id')
    keys = ids.astype(str)
    if pd.Index(keys).has_duplicates:
        raise ValueError('Коллизия ID после преобразования в строковый формат')
    missing = pd.Index(keys).difference(ref.index)
    if len(missing):
        raise ValueError(f'Отсутствуют {len(missing)} ID в справочнике; первые: {missing[:5].tolist()}')
    ref = ref.loc[keys].copy()
    if ref.isna().any().any():
        raise ValueError('Названия и регионы муниципалитетов не могут быть пустыми')
    ref.index = ids
    ref.index.name = ids.name
    return ref


def representative_municipalities(dynamic: pd.DataFrame, mapping: Mapping, aligned: pd.DataFrame,
                                  month: str, municipality_lookup: pd.DataFrame,
                                  annual: pd.DataFrame, top_n: int = 4,
                                  zero_floor: float = 1e-6) -> pd.DataFrame:
    """Ближайшие к CLR-центроиду внутри кластера, независимо от годовых пропусков."""
    if not municipality_lookup.index.equals(dynamic.index):
        raise ValueError('Справочник содержит другой порядок муниципалитетов')
    clr = transform_shares(monthly_view(dynamic,mapping,month), 'clr', zero_floor)
    shares = monthly_view(dynamic,mapping,month)
    context = annual_view(annual,pd.Period(month,freq='M').year)
    lab = aligned[month]
    rows = []
    for cluster, ids in lab.groupby(lab,sort=True).groups.items():
        x = clr.loc[ids].to_numpy(dtype=float)
        center = x.mean(axis=0)
        dist = np.linalg.norm(x-center,axis=1)
        ranked = np.argsort(dist,kind='stable')[:top_n]
        for rank, index_pos in enumerate(ranked,1):
            mid = ids[index_pos]
            row = {'municipality_id':str(mid),
                   'municipality_name':str(municipality_lookup.loc[mid,'municipality_name']),
                   'region':str(municipality_lookup.loc[mid,'region']),
                   'cluster':int(cluster), 'month':str(month), 'rank':rank,
                   'distance_to_clr_centroid':float(dist[index_pos])}
            row.update({k:float(shares.loc[mid,k]) for k in SHARES})
            row.update({k:(float(context.loc[mid,k]) if pd.notna(context.loc[mid,k]) else np.nan)
                        for k in ['wage','population_density','investment_total_per_capita']})
            rows.append(row)
    return pd.DataFrame(rows)


def snapshot_stability(aligned: pd.DataFrame, months: Sequence[str]) -> pd.DataFrame:
    """Доля сохранения экономического типа помесячно по ИСХОДНОМУ типу."""
    rows=[]
    for previous,current in zip(months[:-1],months[1:]):
        groups=aligned[previous].groupby(aligned[previous],sort=True).groups
        for cluster, ids in groups.items():
            share=float((aligned.loc[ids,previous] == aligned.loc[ids,current]).mean())
            rows.append({'month_from':previous,'month_to':current,'cluster':int(cluster),
                         'n_source':int(len(ids)),'stay_share':share,'switch_share':1-share})
    return pd.DataFrame(rows)


def delta_switch_association(aligned: pd.DataFrame, delta: pd.DataFrame,
                             months: Sequence[str], year: int=2024) -> tuple[pd.DataFrame,pd.DataFrame]:
    """Ретроспективная связь годового изменения с числом переходов в год t.

    Никаких причинных интерпретаций и повторной кластеризации.
    """
    pairs=[(a,b) for a,b in zip(months[:-1],months[1:]) if pd.Period(b,freq='M').year==year]
    switches=sum((aligned[a] != aligned[b]).astype(int) for a,b in pairs)
    per_node=pd.DataFrame({'municipality_id':aligned.index.astype(str),
                           'switch_count':switches.to_numpy(dtype=int),
                           'changed_any':(switches>0).to_numpy(dtype=bool)},index=aligned.index)
    rows=[]
    for k in INDICATORS:
        col=f'{k}__delta_{year-1}_{year}'
        vals=delta[col].astype(float)
        valid=vals.notna() & np.isfinite(vals)
        change=vals[valid].to_numpy(float)
        counts=switches[valid].to_numpy(int)
        shifted=counts>0
        rho=float(spearmanr(counts,change).statistic) if len(np.unique(counts))>1 and len(np.unique(change))>1 else np.nan
        rho_abs=float(spearmanr(counts,np.abs(change)).statistic) if len(np.unique(counts))>1 and len(np.unique(np.abs(change)))>1 else np.nan
        rows.append({'indicator':k,'n_available':int(valid.sum()), 'missing_share':float(1-valid.mean()),
                     'n_switchers':int(shifted.sum()),'n_non_switchers':int((~shifted).sum()),
                     'median_delta_switchers':float(np.median(change[shifted])) if shifted.any() else np.nan,
                     'median_delta_non_switchers':float(np.median(change[~shifted])) if (~shifted).any() else np.nan,
                     'spearman_switch_count_delta':rho,'spearman_switch_count_abs_delta':rho_abs,
                     'n_month_pairs':len(pairs)})
    return pd.DataFrame(rows),per_node


def retained_delta_by_type(aligned: pd.DataFrame, delta: pd.DataFrame,
                           left: str='2023-12', right: str='2024-12') -> pd.DataFrame:
    """Только муниципалитеты, сохранившие метку на границах года; наблюдаемые дельты."""
    retained = aligned[left] == aligned[right]
    rows=[]
    for cid, ids in aligned[left].groupby(aligned[left],sort=True).groups.items():
        same=ids[retained.loc[ids].to_numpy()]
        for indicator in INDICATORS:
            v=delta.loc[same,f'{indicator}__delta_2023_2024'].dropna()
            rows.append({'cluster':int(cid),'indicator':indicator,'retained_n':len(same),
                         'n_available':len(v),'median_delta':float(v.median()) if len(v) else np.nan,
                         'q25_delta':float(v.quantile(.25)) if len(v) else np.nan,
                         'q75_delta':float(v.quantile(.75)) if len(v) else np.nan})
    return pd.DataFrame(rows)


def save_tables(directory: str | Path, profiles: pd.DataFrame, interpretations: pd.DataFrame,
                representatives: pd.DataFrame) -> list[Path]:
    """Сохраняет только три небольшие итоговые таблицы."""
    if representatives.empty or not {'municipality_name','region'}.issubset(representatives):
        raise ValueError('Нельзя сохранять таблицу примеров без проверенных названий и регионов')
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    paths=[directory/'economic_profiles.parquet',
           directory/'cluster_interpretations.parquet',
           directory/'typical_municipalities.parquet']
    for path,data in zip(paths,[profiles,interpretations,representatives]):
        data.to_parquet(path,index=False)
    return paths


def market_access_profile(
    market_access: pd.DataFrame,
    aligned: pd.DataFrame,
    month: str = '2024-12',
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ретроспективный профиль индекса доступности рынков СберИндекса (2024).

    Используется только сохранённое разбиение, а не обучение. Пропуски не
    импутируются; соответствие идентификаторов проверяется перед выравниванием.
    """
    if month != '2024-12':
        raise ValueError('Индекс 2024 года сопоставляется только со срезом 2024-12')
    if month not in aligned or not aligned.index.is_unique or aligned.index.hasnans:
        raise ValueError('Отсутствуют проверенные метки или ID муниципалитетов')
    if list(market_access.columns) != ['market_access']:
        raise ValueError('Ожидается подготовленная матрица X_market_access с колонкой market_access')
    if not market_access.index.is_unique or market_access.index.hasnans:
        raise ValueError('Некорректный индекс X_market_access')
    if not pd.api.types.is_numeric_dtype(market_access['market_access']):
        raise ValueError('Индекс доступности рынков должен быть числовым')

    ids = aligned.index.astype(str)
    market_ids = market_access.index.astype(str)
    if not pd.Index(ids).is_unique or not pd.Index(market_ids).is_unique:
        raise ValueError('Коллизия строковых идентификаторов после нормализации')
    unknown = set(market_ids) - set(ids)
    if unknown:
        raise ValueError(f'X_market_access содержит {len(unknown)} посторонних territory_id')

    values = market_access['market_access'].copy()
    values.index = market_ids
    values = values.reindex(ids)
    values.index = aligned.index
    valid = values.dropna().astype(float)
    if valid.empty:
        raise ValueError('Нет наблюдаемых значений market_access для выбранных МО')
    if not np.isfinite(valid.to_numpy()).all():
        raise ValueError('В market_access обнаружены бесконечные значения')
    if valid.lt(-1e-8).any() or valid.gt(1000+1e-8).any():
        raise ValueError('Индекс вне документированного диапазона 0–1000')

    baseline_mean = float(valid.mean())
    baseline_median = float(valid.median())
    baseline_iqr = float(valid.quantile(.75) - valid.quantile(.25))
    labels = aligned[month]
    if labels.isna().any():
        raise ValueError('В выбранном месяце имеются пропуски кластерных меток')
    rows = []
    for cluster, ids_in_type in labels.groupby(labels, sort=True).groups.items():
        v = values.loc[ids_in_type].dropna().astype(float)
        n = len(ids_in_type)
        rows.append({
            'month':month, 'year':2024, 'cluster':int(cluster),
            'family':'market_access_2024', 'indicator':'market_access',
            'cluster_size':int(n), 'observed':int(len(v)), 'coverage':float(len(v)/n),
            'mean':float(v.mean()) if len(v) else np.nan,
            'median':float(v.median()) if len(v) else np.nan,
            'q25':float(v.quantile(.25)) if len(v) else np.nan,
            'q75':float(v.quantile(.75)) if len(v) else np.nan,
            'baseline_mean':baseline_mean, 'baseline_median':baseline_median,
            'baseline_iqr':baseline_iqr,
            'contrast':float((v.median()-baseline_median)/baseline_iqr)
                        if len(v) and baseline_iqr > 0 else np.nan,
            'contrast_unit':'median_difference_over_global_IQR',
        })
    profiles = pd.DataFrame(rows)

    rank = valid.rank(method='average')
    rank_mean = float(rank.mean())
    total_ss = float(((rank - rank_mean)**2).sum())
    weighted_between = 0.0
    for _, idx in labels.loc[valid.index].groupby(labels.loc[valid.index]).groups.items():
        cluster_rank = rank.loc[idx]
        weighted_between += len(cluster_rank) * float(cluster_rank.mean()-rank_mean)**2
    rank_eta_squared = weighted_between/total_ss if total_ss > 0 else np.nan
    diagnostics = pd.DataFrame([{
        'month':month, 'municipalities':int(len(aligned)),
        'observed':int(len(valid)), 'coverage':float(len(valid)/len(aligned)),
        'n_clusters':int(labels.nunique()), 'global_median':baseline_median,
        'global_iqr':baseline_iqr, 'rank_eta_squared':rank_eta_squared,
    }])
    return profiles, diagnostics


def describe_market_access_2024(
    market_profiles: pd.DataFrame,
    market_diagnostics: pd.DataFrame,
    *,
    minimum_coverage: float = 0.80,
) -> tuple[pd.DataFrame, str]:
    """Фактические экономико-географические описания типов за 2024 год.

    Возвращает таблицу с описаниями и Markdown с численными результатами.
    Не меняет имена типов и не делает предположений о скрытых данных.
    """
    mandatory = {'cluster', 'month', 'cluster_size', 'observed', 'coverage',
                 'median', 'q25', 'q75', 'contrast'}
    if not mandatory.issubset(market_profiles.columns):
        raise ValueError('Отсутствуют колонки профилей market_access')
    if market_profiles.empty or market_profiles.cluster.duplicated().any():
        raise ValueError('Ожидается одна строка на экономический тип')
    if market_profiles.month.ne('2024-12').any():
        raise ValueError('Ожидаются профили только за декабрь 2024')
    if len(market_diagnostics) != 1:
        raise ValueError('Ожидается одна запись диагностики market_access')
    for key in ('global_median', 'rank_eta_squared', 'coverage', 'observed', 'municipalities'):
        if key not in market_diagnostics:
            raise ValueError(f'Отсутствует показатель диагностики: {key}')
    if not 0 <= minimum_coverage <= 1:
        raise ValueError('Порог покрытия должен быть в [0, 1]')

    diag = market_diagnostics.iloc[0]
    global_median = float(diag.global_median)
    rows = []
    lines = ['### Фактические результаты по доступности рынков (декабрь 2024)',
             '',
             f'Охват индекса: **{int(diag.observed)} из {int(diag.municipalities)}** '
             f'муниципалитетов (**{100 * float(diag.coverage):.1f}%**). '
             f'Общая медиана: **{global_median:.1f}** (шкала 0–1000).', '']
    for r in market_profiles.sort_values('cluster').itertuples(index=False):
        observed, size, coverage = int(r.observed), int(r.cluster_size), float(r.coverage)
        if observed:
            med = float(r.median)
            if med > global_median:
                direction = 'выше'
            elif med < global_median:
                direction = 'ниже'
            else:
                direction = 'совпадает с'
                # Для равенства используется отдельная грамматическая форма
            diff = med - global_median
            if np.isfinite(float(r.contrast)):
                contrast_text = f'{float(r.contrast):+.2f} общего IQR'
            else:
                contrast_text = 'контраст в IQR не определён'
            relative = ('совпадает с общей медианой' if med == global_median
                        else f'{direction} общей медианы')
            desc = (f'Медиана индекса {med:.1f}, интерквартильный интервал '
                    f'[{float(r.q25):.1f}; {float(r.q75):.1f}]; '
                    f'{relative} (разность {diff:+.1f}; {contrast_text}); '
                    f'покрытие {observed}/{size} ({100*coverage:.1f}%).')
        else:
            desc = f'Нет доступных значений индекса (0/{size}); профиль не определяется.'
        if coverage < minimum_coverage:
            desc += ' Покрытие ниже выбранного порога: вывод требует осторожности.'
        rows.append({'cluster': int(r.cluster), 'market_access_context_2024': desc})
        lines.append(f'- **Тип {int(r.cluster)}.** {desc}')
    lines.append('')

    ranked = market_profiles.loc[market_profiles.observed.gt(0)].sort_values('median')
    if len(ranked) >= 2:
        lo, hi = ranked.iloc[0], ranked.iloc[-1]
        lines.append('**Сравнение типов.** Наименьшая медиана у типа '
                     f'**{int(lo.cluster)}** ({float(lo["median"]):.1f}), '
                     f'наибольшая — у типа **{int(hi.cluster)}** '
                     f'({float(hi["median"]):.1f}); '
                     f'разность составляет **{float(hi["median"]-lo["median"]):.1f}** пункта индекса.')
    eta = float(diag.rank_eta_squared)
    if np.isfinite(eta):
        lines.append(f'Межтиповая доля вариации **рангов** индекса '
                     fr'\(\eta^2_{{\mathrm{{rank}}}}={eta:.3f}\). '
                     'Это мера описательного различия, а не причинный эффект.')
    else:
        lines.append('Ранговая мера межтиповых различий не определена '
                     '(например, при константных значениях индекса).')
    low = market_profiles.loc[market_profiles.coverage.lt(minimum_coverage), 'cluster'].tolist()
    if low:
        lines.append('**Ограничение покрытия:** для типов '
                     + ', '.join(map(lambda x: str(int(x)), low))
                     + ' неполнота наблюдений может влиять на сопоставимость медиан.')
    lines.extend(['',
        '**Интерпретация.** Эти цифры дополняют описание потребительских типов '
        'их экономико-географическим положением. Индекс за 2024 год не '
        'участвовал в обучении, не характеризует фактические потоки и не '
        'доказывает отраслевую специализацию.'])
    return pd.DataFrame(rows), '\n'.join(lines)
