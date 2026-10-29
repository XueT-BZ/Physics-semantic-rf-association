#!/usr/bin/env python3
"""Compose the actual vector Matplotlib figures as a high-resolution review atlas.

No axes are recreated here; original figure PDFs are embedded as vector pages.
"""
from pathlib import Path
import fitz
ROOT=Path(__file__).resolve().parents[1]
PAGES=[
 [('results_pr','Figure 8(a) | Precision-recall curves'),('results_reliability','Figure 8(b) | Reliability with enlarged bin differences')],
 [('results_semantic','Figure 10(a) | Semantic error and signed F1 gap'),('results_clock_f1','Figure 10(b) | Clock error and matched-minus-fixed F1')],
 [('results_clock_gate','Figure 11(a) | Gate recall and uncertainty gap'),('results_runtime','Figure 11(b) | Aligned time panels and percent difference')]
]
notes=[
 'Reliability curves can be nearly coincident by measurement; the independent lower axis expands their actual bin-wise difference.',
 'Upper axes keep true experimental means and confidence intervals. Lower axes display signed differences, not vertically shifted original curves.',
 'Decoder runtimes use two separate plots with identical log scales, followed by the original data-derived percent difference.'
]
pdf=fitz.open()
for idx,(items,note) in enumerate(zip(PAGES,notes),start=1):
 page=pdf.new_page(width=1190.55,height=841.89)
 page.insert_text((40,59),'RF EVENT ASSOCIATION | Experimental chart review',fontsize=23,fontname='hebo',color=(.11,.17,.25))
 page.insert_text((40,87),'Data-preserving, legible comparisons with separately magnified differences',fontsize=13,color=(.35,.40,.45))
 page.draw_line(fitz.Point(40,105),fitz.Point(1150,105),color=(.4,.52,.65),width=1)
 for k,(stem,title) in enumerate(items):
  x=40 if k==0 else 615
  page.insert_text((x,134),title,fontsize=13,fontname='hebo',color=(.1,.19,.3))
  with fitz.open(ROOT/'figures'/f'{stem}.pdf') as fig:
   w=520;h=w*fig[0].rect.height/fig[0].rect.width
   assert h<540
   page.show_pdf_page(fitz.Rect(x,156,x+w,156+h),fig,0)
 page.draw_line(fitz.Point(40,714),fitz.Point(1150,714),color=(.8,.84,.88),width=.8)
 remain=page.insert_textbox(fitz.Rect(40,733,1150,780),note,fontsize=12,lineheight=1.3,color=(.19,.23,.3))
 if remain<0:raise ValueError('Page note overflows')
 page.insert_text((40,805),'All values originate in results/; graph generation: code/make_results_figures.py.',fontsize=10,color=(.45,.48,.52))
 page.insert_text((1110,805),f'{idx} / {len(PAGES)}',fontsize=10)
pdf.set_metadata({'title':'RF event association: source-faithful chart comparison atlas',
                  'subject':'Three large-format figure spreads with data-derived differences'})
pdf.save(ROOT/'experimental_curves_review.pdf',garbage=4,deflate=True)
print('Built three-page A3 vector chart atlas')
