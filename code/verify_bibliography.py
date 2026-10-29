#!/usr/bin/env python3
"""Fast metadata-only audit: check chronology, identity, and citations; no experiments are run."""
from pathlib import Path
import json, csv, hashlib, re
from collections import Counter

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / 'audit' / 'references_2022_2026_review.csv'
ROWS = list(csv.DictReader(DATA.open(encoding='utf-8-sig',newline='')))
assert len(ROWS) == 49
assert len(set(x['key'] for x in ROWS)) == len(ROWS)
assert len(set(x['doi'] for x in ROWS if x['doi'])) == sum(bool(x['doi']) for x in ROWS)
assert all(x['publisher_or_identifier_url'].startswith('https://') for x in ROWS)
years = Counter(int(x['year']) for x in ROWS)
assert sum(2022<=int(x['year'])<=2026 for x in ROWS) == 37
assert sum(x['within_2022_2026']=='1' for x in ROWS) == 37
backslash = chr(92)
source = (BASE/'main.tex').read_text() + '\n' + '\n'.join(p.read_text() for p in sorted((BASE/'sections').glob('*.tex')))
refs = (BASE/'references.tex').read_text()
pat = re.compile(re.escape(backslash)+r'cite(?:p|t)?\{([^}]+)\}')
first=[]
for group in pat.findall(source):
    for k in group.split(','):
        k=k.strip()
        if k not in first:first.append(k)
refpat = re.compile(re.escape(backslash)+r'bibitem\{([^}]+)\}')
bibkeys = refpat.findall(refs)
assert first == bibkeys, 'The bibliography must match the first-citation order.'
assert bibkeys == [r['key'] for r in ROWS], 'CSV must correspond to manuscript numbering.'
assert 'undefined references' not in (BASE/'main.log').read_text()
report={'status':'PASS','reference_count':len(ROWS),'recent_2022_2026':37,
  'recent_share_percent':round(100*37/len(ROWS),2),'year_counts':dict(sorted(years.items())),
  'doi_count':sum(bool(x['doi']) for x in ROWS),
  'publisher_or_identifier_link_count':len(ROWS),
  'all_references_used':True,'reference_order_matches_first_citation':True,
  'source_hash_sha256':hashlib.sha256(refs.encode()).hexdigest(),
  'scope':'References and metadata only; does not rerun or revalidate numerical experiments.'}
(BASE/'audit'/'bibliography_verification.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
print(json.dumps(report,indent=2,ensure_ascii=False))
