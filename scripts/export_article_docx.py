"""Plain editable Word export; all substantive content lives in article_blocks."""
from pathlib import Path
import json,re
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
ROOT=Path(__file__).resolve().parents[1];M=ROOT/'manuscript'
data=json.loads((M/'article_blocks.json').read_text(encoding='utf-8'))
doc=Document();s=doc.sections[0]
s.page_width=Inches(8.27);s.page_height=Inches(11.69)
s.top_margin=s.bottom_margin=Inches(.8);s.left_margin=s.right_margin=Inches(.8)
for name in ['Normal','Title','Heading 1','Heading 2','Caption']:
    st=doc.styles[name];st.font.name='Times New Roman';st.font.color.rgb=RGBColor(0,0,0)
    st.font.size=Pt(11);st.paragraph_format.space_after=Pt(6)
    for border in st._element.xpath('.//w:pBdr'):border.getparent().remove(border)
    for rf in st._element.xpath('.//w:rFonts'):
        for att in list(rf.attrib):
            if 'Theme' in att or 'theme' in att:del rf.attrib[att]
        for namepart in ['ascii','hAnsi','eastAsia','cs']:rf.set(qn('w:'+namepart),'Times New Roman')
doc.styles['Normal'].paragraph_format.line_spacing=1.08
doc.styles['Title'].font.size=Pt(16)
for name in ['Heading 1','Heading 2']:
    doc.styles[name].font.bold=True;doc.styles[name].font.size=Pt(12 if name=='Heading 1' else 11)
    doc.styles[name].paragraph_format.keep_with_next=True
doc.styles['Caption'].font.size=Pt(10)
eqno=0
def mr(text):
    r=OxmlElement('m:r');t=OxmlElement('m:t');t.text=text;r.append(t);return r
def seq(*nodes):return [mr(x) if isinstance(x,str) else x for x in nodes]
def script(base,sub=None,sup=None):
    kind='sSubSup' if sub is not None and sup is not None else ('sSub' if sub is not None else 'sSup')
    node=OxmlElement('m:'+kind);body=OxmlElement('m:e')
    for el in (seq(base) if isinstance(base,str) else base):body.append(el)
    node.append(body)
    for tag,val in [('sub',sub),('sup',sup)]:
        if val is not None:
            s=OxmlElement('m:'+tag);s.append(mr(val));node.append(s)
    return node
def v(x):return script(x,'t')
def mathblock(text):
    global eqno
    eqno+=1
    if text.startswith('Delta G'):
        lines=[seq('ΔG = G(',script('p','DFL'),') − G(',script('p','matched MSE'),').')]
    elif text.startswith('L_DFL'):
        lines=[seq(script('L','DFL'),'(θ) = (1/N) ',script('∑','i'),' ',script('C','i'),'[',script('A','ε'),'(',script('l̂','i'),' − ',script('f','θ'),'(',script('x','i'),'))].')]
    elif text.startswith('e_t'):
        lines=[seq(v('e'),' = ',script('e','t−1'),' + Δ(η ',v('c'),' − ',v('d'),'/η),'),seq('0 ≤ ',v('c'),', ',v('d'),' ≤ 3,'),seq('1 ≤ ',v('e'),' ≤ 9,'),seq(script('e','0'),' = ',script('e','48'),' = 5.')]
    elif text.startswith('J_hat'):
        lines=[seq('Ĵ = Δ ',script('∑','t'),' [',script('p','t','buy'),' max(',v('ĝ'),',0) + ',script('p','t','sell'),' min(',v('ĝ'),',0) + κ(',v('c'),' + ',v('d'),')] + εR(c,d,e),')]
    elif text.startswith('R(c'):
        lines=[seq('R(c,d,e) = ',script('∑','t'),' [',script(seq('(',v('c'),'/3)'),sup='2'),' + ',script(seq('(',v('d'),'/3)'),sup='2'),' + ',script(seq('(',v('e'),'/10)'),sup='2'),'].')]
    else:raise ValueError(text)
    for i,nodes in enumerate(lines):
        p=doc.add_paragraph();p.alignment=1;p.paragraph_format.space_after=Pt(3)
        p.paragraph_format.keep_with_next=i < len(lines)-1
        p.paragraph_format.keep_together=True
        m=OxmlElement('m:oMath')
        for node in nodes:m.append(node)
        p._p.append(m)
        if i==0:p.add_run(f'   ({eqno})')
def add_table(b):
    p=doc.add_paragraph(b['caption'],'Caption');p.paragraph_format.keep_with_next=True
    table=doc.add_table(rows=1,cols=len(b['headers']));table.style='Table Grid'
    for i,h in enumerate(b['headers']):table.rows[0].cells[i].text=h
    pr=table.rows[0]._tr.get_or_add_trPr();repeat=OxmlElement('w:tblHeader');pr.append(repeat)
    for row in b['rows']:
        cells=table.add_row().cells
        for c,v in zip(cells,row):c.text=v
    for ri,row in enumerate(table.rows):
        pr=row._tr.get_or_add_trPr();x=OxmlElement('w:cantSplit');pr.append(x)
        for cell in row.cells:
            for p in cell.paragraphs:
                p.paragraph_format.space_after=Pt(3);p.paragraph_format.space_before=Pt(2)
                p.paragraph_format.line_spacing=1
                for run in p.runs:run.font.size=Pt(9);run.bold=ri==0
    p=doc.add_paragraph(b['note']);p.paragraph_format.space_after=Pt(9)
    for run in p.runs:run.font.size=Pt(9)
for b in data['blocks']:
    kind=b['type']
    if kind=='title':doc.add_paragraph(b['text'],'Title')
    elif kind=='heading':doc.add_paragraph(b['text'],'Heading 2' if re.match(r'^\d\.\d',b['text']) else 'Heading 1')
    elif kind=='table':add_table(b)
    elif kind=='figure':
        p=doc.add_paragraph();p.paragraph_format.keep_with_next=True
        p.add_run().add_picture(b['path'],width=Inches(6.6))
        p=doc.add_paragraph(b['caption'],'Caption');p.paragraph_format.keep_with_next=True
        p=doc.add_paragraph(b['note'])
        for r in p.runs:r.font.size=Pt(9)
    elif kind=='equation':mathblock(b['text'])
    else:
        p=doc.add_paragraph(b['text'])
        if kind=='reference':
            p.paragraph_format.space_after=Pt(5)
            for r in p.runs:r.font.size=Pt(9.5)
footer=s.footer.paragraphs[0];footer.alignment=2
fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');footer._p.append(fld)
doc.core_properties.title=data['title'];doc.core_properties.subject='Original research manuscript for Applied Energy'
doc.core_properties.author='';doc.core_properties.keywords='PV; battery; decision-focused learning; distribution network'
for border in doc._element.xpath('.//w:pBdr'):border.getparent().remove(border)
out=ROOT/'住宅光储决策学习_Applied_Energy英文稿.docx'
doc.save(out)
print(json.dumps({'file':str(out),'paragraphs':len(doc.paragraphs),'tables':len(doc.tables),'math_blocks':eqno,'bytes':out.stat().st_size},ensure_ascii=False))
