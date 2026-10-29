#!/usr/bin/env python3
"""Assemble the seven editable method figures as a vector-PDF review atlas."""
from pathlib import Path
import fitz

ROOT=Path(__file__).resolve().parents[1]
PAGES=[
 [('fig01_task','Figure 1 | Task and failure mechanism',
   'Spatial reception, equivalent native encodings, a receiver-conflicting graph, and a valid candidate partition.'),
  ('fig02_architecture','Figure 2 | Overall architecture',
   'Separate offline fitting and calibration from the online alignment, admission, scoring, and constrained partitioning path.')],
 [('fig03_contract','Figure 3 | Canonical observation contract',
   'Known unit and class mappings preserve values. Shared uncertainty and evaluation-only fields remain separate.'),
  ('fig04_gate','Figure 4 | Physical admission',
   'Geometry and declared uncertainty define a necessary consistency test, not proof of common event identity.'),
  ('fig05_evidence','Figure 5 | Physical and semantic evidence',
   'Five normalized physical descriptors and one concentration-weighted compatibility feature form the six-dimensional input.')],
 [('fig06_temperature','Figure 6 | Calibration and signed-gain scaling',
   'Positive temperature rescales signed evidence. Partition invariance is conditional on unchanged masks, clipping, and decision rules.'),
  ('fig07_decoder','Figure 7 | Feasible merging and the greedy limitation',
   'Receiver disjointness and all-cross-pair admission are mandatory. The four-node illustration has greedy value 5 and optimum 8.')]
]

def put_text(page, rect, text, size=9, font='helv'):
    unused=page.insert_textbox(rect,text,fontsize=size,fontname=font,lineheight=1.15)
    if unused<0:
        raise RuntimeError('Atlas text does not fit: '+text)

def main():
    atlas=fitz.open()
    for page_index,items in enumerate(PAGES,1):
        page=atlas.new_page(width=595.276,height=841.89)
        put_text(page,fitz.Rect(36,27,560,52),'PHYSICS-SEMANTIC RF EVENT ASSOCIATION',14,'hebo')
        put_text(page,fitz.Rect(36,53,560,71),'Editable method-figure atlas | Companion to the revised manuscript',9)
        page.draw_line(fitz.Point(36,78),fitz.Point(560,78),width=.6)
        y=92
        for stem,title,caption in items:
            put_text(page,fitz.Rect(36,y,560,y+22),title,11,'hebo');y+=22 if page_index==2 else 26
            drawing_width=510 if page_index==2 else 524
            drawing_left=(595.276-drawing_width)/2
            with fitz.open(ROOT/'figures'/(stem+'.pdf')) as source:
                height=drawing_width*source[0].rect.height/source[0].rect.width
                page.show_pdf_page(fitz.Rect(drawing_left,y,drawing_left+drawing_width,y+height),source,0)
            y+=height+4
            put_text(page,fitz.Rect(36,y,560,y+27),caption,8.7 if page_index==2 else 9);y+=31 if page_index==2 else 42
        if y>798:
            raise RuntimeError(f'Atlas content reaches footer on page {page_index}: {y}')
        if page_index==3:
            y+=12
            page.draw_line(fitz.Point(36,y),fitz.Point(560,y),width=.5);y+=17
            put_text(page,fitz.Rect(36,y,560,y+22),'Reading and editing these figures',11,'hebo');y+=27
            note=(
              'These are conceptual and algebraic illustrations, not measured RF results. '
              'The performance plots in the manuscript are regenerated separately from the executed experiment outputs.\n\n'
              'All seven figure sources are in figure_sources/fig*.tex; their generator is code/make_diagrams.py. '
              'The delivered files are vector PDFs with editable text and geometry, accompanied by PNG previews.\n\n'
              'Independent checks cover unit conversions, native class permutations, uniform-semantics neutrality, '
              'temperature scaling, receiver exclusivity, and all 15 partitions of the four-node counterexample.'
            )
            put_text(page,fitz.Rect(36,y,560,782),note,10)
        put_text(page,fitz.Rect(36,807,515,825),'Conceptual figures only | Full captions and mathematical conditions are in main.pdf.',8)
        put_text(page,fitz.Rect(533,807,560,825),str(page_index)+' / 3',8)
    atlas.set_metadata({'title':'Physics-Semantic RF Event Association: Method-Figure Atlas',
                        'subject':'Seven editable conceptual figures accompanying the revised manuscript',
                        'author':'Research manuscript - author information pending'})
    atlas.save(ROOT/'figures_overview.pdf',garbage=4,deflate=True)
    print('Created figures_overview.pdf (3 pages).')
if __name__=='__main__':main()
