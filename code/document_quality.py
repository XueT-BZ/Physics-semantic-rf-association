#!/usr/bin/env python3
"""Check document completeness, page boundaries, and build warnings.
This complements, but does not replace, visual inspection of every rendered page.
"""
from pathlib import Path
import hashlib
import json
import fitz

ROOT=Path(__file__).resolve().parents[1]

def main():
    checks=[];outside=[]
    paths=[ROOT/'main.pdf',ROOT/'supplement.pdf']+sorted((ROOT/'figures').glob('*.pdf'))
    if (ROOT/'figures_overview.pdf').exists():paths.append(ROOT/'figures_overview.pdf')
    for path in paths:
        doc=fitz.open(path);span_count=0
        for page_num,page in enumerate(doc,1):
            for block in page.get_text('dict')['blocks']:
                if 'lines' not in block:continue
                for line in block['lines']:
                    for span in line['spans']:
                        if not span['text'].strip():continue
                        span_count+=1
                        r=fitz.Rect(span['bbox']);r0=page.rect
                        if r.x0 < r0.x0-.8 or r.y0 < r0.y0-.8 or r.x1 > r0.x1+.8 or r.y1 > r0.y1+.8:
                            outside.append({'file':str(path.relative_to(ROOT)),'page':page_num,
                                            'text':span['text'],'bbox':list(r)})
        checks.append({'file':str(path.relative_to(ROOT)),'pages':len(doc),'text_spans':span_count,
                       'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    warnings=[]
    for path in [ROOT/'main.log',ROOT/'supplement.log']+sorted((ROOT/'figure_sources').glob('fig*.log')):
        txt=path.read_text(errors='replace')
        for keyword in ['Overfull','undefined references','undefined citations','LaTeX Error','Fatal error']:
            if keyword in txt:warnings.append({'file':str(path.relative_to(ROOT)),'keyword':keyword})
    report={'status':'PASS' if not outside and not warnings else 'REVIEW',
            'pdf_files':checks,'page_boundary_issues':outside,'build_issues':warnings,
            'scope':'Text page-boundary and selected build-warning checks. Node containment, arrows, and visual hierarchy were additionally inspected on rendered images.'}
    (ROOT/'audit'/'document_quality_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='pdf_files'},indent=2))
    if outside or warnings:raise AssertionError('Inspect document_quality_report.json before release.')
if __name__=='__main__':main()
