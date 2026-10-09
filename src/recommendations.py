"""A full-source catalogue and an explicit analyst-reviewed acquisition plan."""
from pathlib import Path
import json
import math
import pandas as pd
from .local_exports import code_key


def read_plan(root, config):
    return json.loads((Path(root) / config.get('indicator_plan', 'configs/indicator_plan.json')).read_text(encoding='utf-8'))


def full_catalog(reference, plan=(), evidence=None, previous_codes=()):
    rules = {code_key(row['indicator_code']): row for row in plan}
    rows = []
    for source in reference:
        code = code_key(source['_indicator_code'])
        if code in {'', 'nan', 'None'}:
            raise ValueError('В полном справочнике есть показатель без кода')
        rule = rules.get(code, {})
        row = dict(indicator_code=code, indicator_name=source['_indicator_name'],
                   section=int(source['_section']), section_title=source.get('Раздел', ''),
                   unit=source.get('_unit', ''), frequency=source.get('Периодичность', ''),
                   available_years=source.get('Годы', ''), dimensions=source.get('Специальные атрибуты', ''),
                   download_url=source.get('Скачать'),
                   source_hint_2023=source.get('2023.0'), source_hint_2024=source.get('2024.0'),
                   in_previous_shortlist=code in previous_codes,
                   feature_key=f'indicator_{code}', view=f'section_{int(source["_section"])}',
                   match_score=0, priority=10000, load_default=False)
        row.update(rule)
        has23 = str(row['source_hint_2023']) not in {'—', '', 'nan', 'None'}
        has24 = str(row['source_hint_2024']) not in {'—', '', 'nan', 'None'}
        if rule:
            row['decision'] = rule['role']
        elif not has23 and not has24:
            row['decision'] = 'no_focus_data'
            row['why'] = 'В справочнике нет данных за 2023 и 2024; не приоритет для выбранного периода.'
        elif not (has23 and has24):
            row['decision'] = 'single_year_context'
            row['why'] = 'Нет обоих фокусных лет; только отдельный контекст без межгодовой динамики.'
        else:
            row['decision'] = 'not_prioritized'
            row['why'] = 'Доступен для поиска; не вошёл в компактный план по содержательной независимости и сложности разрезов.'
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.indicator_code.duplicated().any():
        raise ValueError('Код показателя неуникален; каталог требует уточнения')
    if evidence is not None and not evidence.empty:
        fields = ['indicator_code', 'base_coverage_2023', 'base_coverage_2024', 'paired_coverage', 'conflicts']
        local = evidence[fields].copy()
        local.indicator_code = local.indicator_code.map(code_key)
        result = result.merge(local, on='indicator_code', how='left', validate='one_to_one')
    return result.sort_values(['priority', 'section', 'indicator_code']).reset_index(drop=True)


def recommended_queue(catalog, include_optional=False):
    mask = catalog['load_default'].eq(True)
    if include_optional:
        mask |= catalog['role'].isin(['candidate', 'alternative', 'support_optional'])
    return catalog[mask].sort_values('priority').reset_index(drop=True)


def search_catalog(catalog, text='', section=None):
    result = catalog
    if text:
        result = result[result.indicator_name.str.contains(text, case=False, regex=False, na=False)
                        | result.indicator_code.eq(text)]
    if section is not None:
        result = result[result.section.eq(int(section))]
    return result
