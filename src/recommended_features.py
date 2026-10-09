"""Unit-aware annual context. Preserve every base object and every missing value."""
import numpy as np
import pandas as pd

SCALE = {'рубль': 1., 'тысяча рублей': 1000., 'человек': 1., 'единица': 1.,
         'процент': 1., 'гектар': .01, 'квадратный метр': 1.,
         'квадратный метр общей площади': 1., 'тысяча квадратных метров': 1000.,
         'километр, тысяча метров': 1., 'тысяча кубических метров': 1000.}


def build_recommended_features(data, mapping, ids, plan, years=(2023, 2024), minimum_coverage=.9):
    """Raw helper values -> unit conversion -> reviewed ratios -> measured inclusion.

    Returns all candidate levels, eligible core features, coverage and dictionary.
    No imputation; no empirical feature selection from future outcomes.
    """
    grid = mapping[mapping.year.isin(years)].copy()
    index = pd.MultiIndex.from_product([ids, years], names=['territory_id', 'year'])
    wide = pd.DataFrame(index=index)
    definitions = {}
    for item in plan:
        key = item['feature_key']
        local = data[data.feature_key.eq(key) & data.view.eq(item['view'])].copy()
        if local.empty:
            continue
        if local.duplicated(['oktmo', 'year']).any():
            raise ValueError(f'Неуникальный показатель: {key}')
        units = local.loc[local.value.notna(), 'unit'].fillna('').str.strip().str.casefold()
        if not units.isin(SCALE).all() or units.nunique() != 1:
            raise ValueError(f'Непроверенные или смешанные единицы {key}: {units.unique()}')
        local['value'] = local['value'] * local.unit.str.strip().str.casefold().map(SCALE)
        joined = grid.merge(local[['oktmo','year','value']], on=['oktmo','year'], how='left', validate='many_to_one')
        wide[key] = joined.set_index(['territory_id','year']).value.reindex(index)
        definitions[key] = item
    features = pd.DataFrame(index=index)
    dictionary = []
    def add(name, value, role, source, interpretation, view):
        features[name] = value.replace([np.inf, -np.inf], np.nan)
        dictionary.append(dict(feature=name, role=role, view=view, source=source, definition=interpretation))
    for key, item in definitions.items():
        mode=item['transformation']; role=item['role']; value=wide[key]
        if mode in {'population','area','numerator','denominator'}:
            continue
        if mode in {'per_capita','per_1000'}:
            denominator = item.get('denominator', 'population_mean')
            if denominator not in wide:
                value = pd.Series(np.nan, index=index)
            else:
                value = value / wide[denominator].where(wide[denominator].gt(0))
            factor = 1000. if mode == 'per_1000' else 1.
            add(f'{key}_{mode}', value * factor, role, item['indicator_code'],
                f'После конверсии единиц / {denominator} × {factor:g}', item['view'])
        else:
            if mode == 'percent':
                value = value.where(value.between(0, 100))
            add(key, value, role, item['indicator_code'], item['why'], item['view'])
    if {'population_jan','land_area_ha'} <= set(wide):
        add('population_density', wide.population_jan.where(wide.population_jan.gt(0)) / wide.land_area_ha.where(wide.land_area_ha.gt(0)),
            'primary','8112027 / 8006001','Человек на км²: население на 1 января / (площадь в га × 0.01)','demography')
    if {'lit_street_km','street_km'} <= set(wide):
        ratio=wide.lit_street_km / wide.street_km.where(wide.street_km.gt(0)) * 100
        add('lit_street_share',ratio.where(ratio.between(0,100)), 'primary','8006003 / 8006007',
            'Доля освещённой протяжённости улиц, %; вне [0,100] исключено','infrastructure')
    if {'equipped_housing','housing_stock'} <= set(wide):
        ratio=wide.equipped_housing / wide.housing_stock.where(wide.housing_stock.gt(0)) * 100
        add('equipped_housing_share',ratio.where(ratio.between(0,100)),'candidate','9010004 / 8008019',
            'Доля благоустроенной жилой площади, %; одинаковый масштаб м²','utilities')
    reports=[]
    dictionary=pd.DataFrame(dictionary)
    for feature in features:
        values=features[feature].unstack('year').reindex(ids)
        row=dict(feature=feature, paired_coverage=float(values[list(years)].notna().all(axis=1).mean()))
        row.update({f'coverage_{year}':float(values[year].notna().mean()) for year in years})
        row['minimum_coverage']=min(row[f'coverage_{year}'] for year in years)
        role=dictionary.loc[dictionary.feature.eq(feature),'role'].iloc[0]
        row['core_eligible']=role=='primary' and row['minimum_coverage']>=minimum_coverage and row['paired_coverage']>=minimum_coverage
        reports.append(row)
    coverage=pd.DataFrame(reports)
    if coverage.empty:
        return features, features.copy(), coverage, dictionary
    core=list(coverage.loc[coverage.core_eligible,'feature'])
    return features, features[core], coverage, dictionary


def flatten_annual(levels, ids, years=(2023,2024)):
    """Two observed levels; differences exported separately to avoid triple weighting."""
    parts=[]
    for year in years:
        part=levels.xs(year,level='year').reindex(ids)
        part=part.rename(columns=lambda name:f'{name}__{year}')
        parts.append(part)
    matrix=pd.concat(parts,axis=1).reindex(ids)
    delta=levels.xs(years[-1],level='year').reindex(ids)-levels.xs(years[0],level='year').reindex(ids)
    delta.columns=[f'{name}__delta_{years[0]}_{years[-1]}' for name in delta]
    return matrix,delta


def redundancy_report(levels, year=2024, threshold=.85, min_pairs=500):
    """Report redundant pairs without silently dropping contentually different features."""
    snapshot=levels.xs(year,level='year')
    correlations=snapshot.corr(method='spearman',min_periods=min_pairs)
    rows=[]
    for i,left in enumerate(snapshot.columns):
        for right in snapshot.columns[i+1:]:
            rho=correlations.loc[left,right]
            if pd.notna(rho) and abs(rho)>=threshold:
                rows.append(dict(left=left,right=right,spearman=float(rho),
                                 observed_pairs=int(snapshot[[left,right]].notna().all(axis=1).sum()),year=year))
    return pd.DataFrame(rows,columns=['left','right','spearman','observed_pairs','year'])
