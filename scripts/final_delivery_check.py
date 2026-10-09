from pathlib import Path
import json, re, hashlib
from docx import Document
from PIL import Image, ImageOps, ImageDraw
from pypdf import PdfReader

root = Path(__file__).resolve().parents[1]
m = root / 'manuscript'
data = json.loads((m/'article_blocks.json').read_text(encoding='utf-8'))
docpath = root/'住宅光储决策学习_Applied_Energy英文稿.docx'
doc = Document(docpath)
assert len(doc.tables) == 4
assert len(doc.inline_shapes) == 2
assert len(data['reference_map']) == 33
text = (m/'article_full.txt').read_text(encoding='utf-8')
assert not re.search(r'\b(untouched|TODO|TBD)\b', text)
assert 'below 3.01×10^-12' in text and 'below 3.08×10^-7' in text
assert 'Under the fixed seed-11 models, the geographically narrower' in text
assert 'Supplementary comparisons against frozen MSE' in text
render = root/'temp'/'article_render_final'
pdfs = list(render.glob('*.pdf'))
assert len(pdfs) == 1
pdf = PdfReader(str(pdfs[0]))
pages = len(pdf.pages)
assert all(len(p.extract_text().strip()) > 100 for p in pdf.pages)
for start in range(1, pages+1, 4):
    sheet = Image.new('RGB', (1414, 2040), 'white')
    draw = ImageDraw.Draw(sheet)
    for k,n in enumerate(range(start, min(start+4, pages+1))):
        im = Image.open(render/f'page-{n}.png').convert('RGB')
        im.thumbnail((707, 1000))
        x,y=(k%2)*707,(k//2)*1020
        sheet.paste(im,(x,y+20));draw.text((x+15,y+2), f'Page {n}', fill='black')
    sheet.save(render/f'review-{start}.jpg', quality=90)
report = {
 'stage':'Completed research manuscript; author submission metadata and public computational archive remain separate pre-submission actions',
 'decision':'Manuscript delivery GO; no acceptance guarantee or completed-submission claim',
 'numerical_review':'PASS after seed-scope and upward-rounded residual corrections',
 'content_review':'PASS after frozen-MSE sensitivity comparator, simulation wording and household-held-out clarification',
 'network_review':'PASS; 280 outcome records independently recomputed',
 'document':{'path':str(docpath),'bytes':docpath.stat().st_size,'pages':pages,'tables':4,'figures':2,'references':33,
             'sha256':hashlib.sha256(docpath.read_bytes()).hexdigest()},
 'source_sha256':hashlib.sha256((m/'article_full.txt').read_bytes()).hexdigest(),
 'no_new_experiment_in_final_revision':True,
}
(root/'research'/'manuscript_delivery_check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
