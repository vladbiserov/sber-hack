"""Диагностика признаковых пространств (этап 05). Без построения графов."""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd

SHARES = ("share_food", "share_health", "share_marketplace", "share_dining", "share_transport", "share_other")


def locate_feature_files(root: Path, bdmo_config: dict) -> dict[str, Path]:
    directory = root / bdmo_config["output_dir"] / "features"
    def choose(exact: str | None, pattern: str) -> Path:
        if exact and (directory / exact).is_file():
            return directory / exact
        matches = sorted(p for p in directory.glob(pattern) if p.is_file())
        if len(matches) != 1:
            raise FileNotFoundError(f"Ожидается один файл {directory / pattern}, найдено {len(matches)}: {matches}")
        return matches[0]
    return {"dynamic": choose(None, '*dynamic*.parquet'),
            "annual": choose('X_annual_core_raw.parquet', 'X_annual*raw.parquet'),
            "delta": choose('X_annual_core_delta.parquet', '*delta.parquet')}


def _parse_column(column: object, categories: tuple[str, ...]) -> tuple[pd.Period, str]:
    parts = column if isinstance(column, tuple) else (column,)
    s = ' '.join(map(str, parts))
    dates = re.findall(r'(?<!\d)(20\d{2})[-_/ .]?(0[1-9]|1[0-2])(?!\d)', s)
    if len(dates) != 1:
        raise ValueError(f"Не удалось однозначно определить месяц в колонке {column!r}; форматы необходимо проверить")
    found = [c for c in categories if re.search(r'(?<![a-zA-Z])'+re.escape(c)+r'(?![a-zA-Z])', s)]
    if len(found) != 1:
        raise ValueError(f"Не удалось однозначно определить категорию в колонке {column!r}: {found}")
    return pd.Period(f'{dates[0][0]}-{dates[0][1]}',freq='M'), found[0]


def parse_dynamic_columns(frame: pd.DataFrame, categories: tuple[str, ...] = SHARES) -> dict[tuple[pd.Period,str], object]:
    """Однозначная карта (месяц, категория) -> исходное имя без перестановки строк."""
    mapping = {}
    for col in frame.columns:
        key = _parse_column(col, categories)
        if key in mapping: raise ValueError(f"Дублируется колонка {key}")
        mapping[key] = col
    months = sorted({m for m, _ in mapping})
    missing = [(m,c) for m in months for c in categories if (m,c) not in mapping]
    if missing: raise ValueError(f"Неполные месяцы: {missing[:8]}")
    if len(months)!=24 or len(mapping)!=144:
        raise ValueError(f"Ожидалось 24×6=144, найдено {len(months)} месяцев и {len(mapping)} колонок")
    if any(months[i+1] != months[i]+1 for i in range(len(months)-1)):
        raise ValueError('Временной ряд содержит пропуск месяца')
    return mapping


def monthly_view(frame: pd.DataFrame, mapping: dict, month: str | pd.Period) -> pd.DataFrame:
    """Только указанный месяц; гарантированно сохраняет индекс муниципалитетов."""
    p = pd.Period(month, freq='M')
    return frame.loc[:,[mapping[p,c] for c in SHARES]].set_axis(SHARES,axis=1,copy=False)


def window_view(frame: pd.DataFrame, mapping: dict, end: str | pd.Period, length: int, representation: str = 'raw', zero_floor: float = 1e-6) -> pd.DataFrame:
    """Окно из length месяцев с концом end (без будущих наблюдений)."""
    if length<1: raise ValueError('length должно быть положительным')
    end = pd.Period(end, freq='M')
    periods = [end-(length-1-i) for i in range(length)]
    missing = [p for p in periods if (p,SHARES[0]) not in mapping]
    if missing: raise ValueError(f'Месяцев нет в доступных данных: {missing}')
    frames = [transform_shares(monthly_view(frame,mapping,p),representation,zero_floor).add_prefix(str(p)+'__') for p in periods]
    return pd.concat(frames,axis=1)


def composition_diagnostics(frame: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    results = []
    for p in sorted({m for m,_ in mapping}):
        x = monthly_view(frame,mapping,p)
        values=x.to_numpy(dtype=float)
        sums=x.sum(axis=1,min_count=len(SHARES))
        results.append({'month':str(p),'missing_cells':int(x.isna().sum().sum()),'zero_cells':int((x==0).sum().sum()),
                        'negative_cells':int((x<0).sum().sum()),'over_one_cells':int((x>1).sum().sum()),
                        'nonfinite_cells':int((~np.isfinite(values)).sum()),
                        'max_sum_error':float((sums-1).abs().max()),'min_share':float(np.nanmin(values)),
                        'max_share':float(np.nanmax(values))})
    return pd.DataFrame(results).set_index('month')


def transform_shares(x: pd.DataFrame, representation: str='raw', zero_floor: float=1e-6) -> pd.DataFrame:
    """Raw либо CLR по строкам; нули замещаются floor и композиция замыкается на единицу.

    Сдвиг floor для нулей — регуляризация, не оценка реальных нулевых расходов.
    Нулевая и отсутствующая строка не восстанавливается.
    """
    a=x.to_numpy(dtype=float)
    if not np.isfinite(a).all() or (a<0).any(): raise ValueError('CLR требует конечных неотрицательных долей без пропусков')
    if not np.allclose(a.sum(axis=1),1,atol=1e-6): raise ValueError('Сумма долей должна равняться 1')
    if representation=='raw': return x.copy()
    if representation!='clr': raise ValueError('representation: raw или clr')
    if not (0<zero_floor<1/len(SHARES)): raise ValueError('Недопустимый zero_floor')
    z=a.copy()
    for i in range(len(z)):
        zeros=z[i]==0
        k=int(zeros.sum())
        if k:
            if k==len(SHARES):raise ValueError('Вся композиция нулевая')
            z[i,~zeros]*=(1-k*zero_floor)/z[i,~zeros].sum()
            z[i,zeros]=zero_floor
    logs=np.log(z)
    clr=logs-logs.mean(axis=1,keepdims=True)
    return pd.DataFrame(clr,index=x.index,columns=x.columns)


def sampled_distances(x: pd.DataFrame, y: pd.DataFrame, sample_pairs:int=12000, random_state:int=42) -> pd.DataFrame:
    if not x.index.equals(y.index):raise ValueError('Индексы не совпадают')
    n=len(x)
    if n<2:raise ValueError('Недостаточно МО')
    rng=np.random.default_rng(random_state)
    i=rng.integers(0,n,size=sample_pairs);j=rng.integers(0,n-1,size=sample_pairs)
    j+=(j>=i)
    a=x.to_numpy(dtype=float);b=y.to_numpy(dtype=float)
    return pd.DataFrame({'raw_distance':np.linalg.norm(a[i]-a[j],axis=1),
                         'clr_distance':np.linalg.norm(b[i]-b[j],axis=1)})


def annual_summary(x: pd.DataFrame) -> pd.DataFrame:
    a=x.apply(pd.to_numeric, errors='raise')
    q1=a.quantile(.25);q3=a.quantile(.75);iqr=q3-q1
    return pd.DataFrame({'missing_fraction':a.isna().mean(),'min':a.min(),'p25':q1,'median':a.median(),
                         'p75':q3,'max':a.max(),'iqr_outlier_fraction':((a.lt(q1-1.5*iqr)|a.gt(q3+1.5*iqr))&a.notna()).mean(),
                         'skew':a.skew()})


def verify_index(*frames:pd.DataFrame)->None:
    baseline=frames[0].index
    if not baseline.is_unique or baseline.hasnans: raise ValueError('Недопустимый индекс первого набора')
    for x in frames[1:]:
        if not x.index.equals(baseline):raise ValueError('Порядок или ID муниципалитетов не совпадают')


def infer_annual_columns(columns:pd.Index)->pd.DataFrame:
    out=[]
    for c in columns:
        s=' '.join(map(str,c)) if isinstance(c,tuple) else str(c)
        years=re.findall(r'(?<!\d)(20\d{2})(?!\d)',s)
        out.append({'column':str(c),'year':years[0] if len(years)==1 else 'НЕ РАСПОЗНАН',
                    'base_indicator':re.sub(r'(?<!\d)20\d{2}(?!\d)','',s).strip(' _-()')})
    return pd.DataFrame(out)
