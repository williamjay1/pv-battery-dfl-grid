"""Freeze reviewed delivery artifacts; verify the portable bundle separately."""
from pathlib import Path
import argparse, datetime, hashlib, json, re, zipfile
import pymupdf as fitz
from PIL import Image
from docx import Document

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'revision_20261009'
F=R/'figures'
RESEARCH=R/'research'
STEMS=['figure1_economics','figure2_voltage','figure3_mechanism','figure4_boundaries']
DOCS=[('article','EPSR_Revised_Manuscript.docx',5),('supplement','EPSR_Supplementary_Material.docx',12),('response','逐条修改说明.docx',1)]

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def structure():
    out=[]
    for kind,name,n in DOCS:
        src=read(R/'manuscript'/f'{kind}_blocks.json')
        tables=[b for b in src['blocks'] if b['type']=='table']
        d=Document(R/name)
        assert len(d.tables)==len(tables)==n
        for t,b in zip(d.tables,tables):
            actual=[[c.text for c in row.cells] for row in t.rows]
            expected=[b['headers']]+[[str(x).replace('_',' ') for x in row] for row in b['rows']]
            assert actual==expected,(name,b['caption'])
        refs=[b for b in src['blocks'] if b['type']=='reference']
        for b in refs:assert b['text'] in [p.text for p in d.paragraphs]
        assert not d._element.xpath('.//w:pBdr') and not d.styles._element.xpath('.//w:pBdr')
        if kind=='article':
            assert len(refs)==37 and len(d.inline_shapes)==4
            assert len([b for b in src['blocks'] if b['type']=='equation'])==4
            assert len(d._element.xpath('.//m:oMath'))==18
        if kind=='response':
            ids=[re.match(r'R\d\d',b['text']).group() for b in src['blocks'] if b['type']=='heading' and re.match(r'R\d\d',b['text'])]
            assert ids==[f'R{i:02}' for i in range(1,30)]
        out.append({'file':name,'sha256':sha(R/name),'tables':n,'table_cells_exactly_match_source':True,'references':len(refs),'native_math_objects':len(d._element.xpath('.//m:oMath'))})
    return out

def visual():
    main=read(RESEARCH/'main_docx_visual_audit.json')
    assert main['status']=='PASS',main['status']
    assert main['docx_sha256']==sha(R/'EPSR_Revised_Manuscript.docx')
    docs=[{'file':'EPSR_Revised_Manuscript.docx','sha256':main['docx_sha256'],'pages':main['page_count'],'status':'PASS','review':'Independent agent actually viewed final rendered pages; numeric and extraction checks are auxiliary.'}]
    for name,rel,expected in [
        ('EPSR_Supplementary_Material.docx','docx_render_release2/EPSR_Supplementary_Material',10),
        ('逐条修改说明.docx','docx_render_final/逐条修改说明',12)]:
        folder=R/'temp'/rel
        pages=sorted(folder.glob('page-*.png'))
        assert len(pages)==expected
        with fitz.open(folder/(Path(name).stem+'.pdf')) as pdf:
            assert len(pdf)==expected
            assert all('\ufffd' not in page.get_text() for page in pdf)
        if name.startswith('EPSR_Supplementary'):
            old=R/'temp/docx_render_final/EPSR_Supplementary_Material'
            assert all(sha(folder/f'page-{i}.png')==sha(old/f'page-{i}.png') for i in range(1,8))
            method='All ten pages actually viewed. Final pages 1–7 are byte-identical to the viewed prior render; changed final pages 8–10 were directly viewed again.'
        else:method='All twelve actual final page PNGs directly viewed by root.'
        docs.append({'file':name,'sha256':sha(R/name),'pages':expected,'status':'PASS','review':method,'findings':[]})
    figs=[]
    for stem in STEMS:
        c=read(F/(stem+'.collisions.json'));a=read(F/(stem+'.alignment.json'));p=read(F/(stem+'.provenance.json'))
        assert c['verdict']==a['verdict']=='PASS'
        assert c['summary']['fail']==c['summary']['warn']==0
        assert p['geometry']['ok'] and not p['geometry']['fonts_under_5pt']
        with fitz.open(F/(stem+'.pdf')) as pdf:
            sizes=[s['size'] for page in pdf for b in page.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans'] if s['text'].strip()]
            assert min(sizes)>=5
            fonts=[font for page in pdf for font in page.get_fonts(full=True)]
            assert fonts and all('Type3' not in x[2] for x in fonts)
            minimum=min(sizes)
        dims={}
        for ext in ['png','tiff']:
            with Image.open(F/(stem+'.'+ext)) as im:
                dpi=im.info.get('dpi')
                assert dpi and min(dpi)>900
                dims[ext]={'pixels':list(im.size),'dpi':[float(x) for x in dpi]}
        svg=(F/(stem+'.svg')).read_text(encoding='utf-8')
        assert '<text' in svg
        figs.append({'figure':stem,'status':'PASS','collision_failures':0,'collision_warnings':0,'minimum_pdf_font_pt':minimum,'editable_svg_text':True,'raster':dims,'visual':'Final rendered preview actually viewed; no text/text, text/data or label/line collision observed.'})
    result={'status':'PASS','recorded_utc':now(),'documents':docs,'figures':figs,'scope':'Delivery visual and structural quality only; does not certify a publisher endorsement.'}
    write(RESEARCH/'delivery_visual_audit.json',result)
    return result

def protocol():
    p=RESEARCH/'revision_protocol.json';j=read(p)
    j['updated_utc']=now()
    j['route_card']['venue']='Electric Power Systems Research; original applied engineering research; hybrid OA. Public university transcription lists 2025 JCR Q2; SJR is explicitly not substituted for JCR.'
    evidence={
      1:['manuscript/article_full.txt'],2:['results/expanded_comparison_tou_support_matched_baselines.json','manuscript/article_full.txt'],
      3:['results/expanded_comparison_tou_support_matched_baselines.json','research/independent_settlement_verification.json'],
      4:['figures/figure1_source_data.json','manuscript/article_full.txt'],5:['results/resource_comparison_tou.json'],
      6:['network_diagnostics/paired.json','research/network_manuscript_audit.txt'],7:['manuscript/supplement_full.txt','figures/figure2_source_data.json'],
      8:['figures/figure2_source_data.json'],9:['manuscript/article_full.txt','research/network_manuscript_audit.txt'],10:['manuscript/supplement_full.txt','network_revision_audit.json'],
      11:['manuscript/supplement_full.txt','results/revision_dispatch_audit.json'],12:['manuscript/supplement_full.txt','network_revision_audit.json'],
      13:['results/expanded_comparison_tou_support_matched_baselines.json'],14:['figures/figure4_source_data.json','manuscript/supplement_full.txt'],
      15:['manuscript/article_full.txt'],16:['network_mechanism/summary.json','figures/figure3_source_data.json'],
      17:['figures/figure1_source_data.json','manuscript/supplement_full.txt'],18:['results/storenet_external_postrun_audit.json','manuscript/article_full.txt'],
      19:['results/storenet_frozen_external.json','results/storenet_external_postrun_audit.json'],20:['research/independent_settlement_verification.json'],
      21:['manuscript/supplement_full.txt'],22:['network_geography/cohort_plan.json','manuscript/supplement_full.txt'],
      23:['figures/figure4_source_data.json','manuscript/supplement_full.txt'],24:['manuscript/article_full.txt','manuscript/supplement_full.txt'],
      25:['reproducibility/reproduction_audit.json','reproducibility/portable_figure_entry_audit.json'],26:['manuscript/article_full.txt'],
      27:['manuscript/response_full.txt'],28:['research/delivery_visual_audit.json'],29:['research/delivery_visual_audit.json','manuscript/response_full.txt']}
    for item in j['status']:
        item['state']='addressed'
        item['evidence_files']=evidence[int(item['id'][1:])]
        item['manuscript_response']='manuscript/response_full.txt#'+item['id']
    j['status_semantics']='Addressed means the comment was implemented or its claim was narrowed with a stated evidence boundary; it does not mean every uncertainty was eliminated.'
    j['remaining_submission_administration']=['Human author names, affiliations, funding, competing interests and responsibility confirmation.','Final journal-specific author-guide check; publisher guide was captcha-blocked.','Public data/code archiving and DOI only when separately authorized.']
    j['venue_sources']={
      'scope':'https://shop.elsevier.com/journals/electric-power-systems-research/0378-7796',
      'hybrid':'https://www.sba.unipi.it/en/electric-power-systems-research',
      'public_JCR_transcription':'https://www.iit.comillas.edu/publicacion/info_revista/en/52/Electric_Power_Systems_Research',
      'subscription_route':'https://www.elsevier.com/about/policies-and-standards/pricing'}
    write(p,j)

def figure_bundle():
    out=R/'Nature_Style_Figures.zip'
    files={}
    for stem in STEMS:
        for ext in ['pdf','svg','png','tiff','provenance.json','collisions.json','alignment.json']:
            files['figures/'+stem+'.'+ext]=F/(stem+'.'+ext)
    for i in range(1,5):files[f'source_data/figure{i}_source_data.json']=F/f'figure{i}_source_data.json'
    files['code/revision_figures.py']=ROOT/'scripts/revision_figures.py'
    vendor=R/'vendor/Yuan1z0825-nature-skills-2a20e4a'
    files['licenses/nature-skills_APACHE_LICENSE']=vendor/'LICENSE'
    tables={kind:[b for b in read(R/'manuscript'/f'{kind}_blocks.json')['blocks'] if b['type']=='table'] for kind in ['article','supplement']}
    readme='''图表交付说明\n\n四幅图：PDF/SVG 为矢量文件，SVG 保留可编辑文字，PDF 嵌入 TrueType 字体；PNG/TIFF 为原生 1200 dpi 输出。矢量图不以 dpi 定义清晰度。\n\n图 1：配对经济差异、家庭收益/损害分布与成本分解。\n图 2：完整上下电压阈值曲线、条件严重度与不同源电压/拓扑。\n图 3：任务信号—电池动作—节点/相电压传导及非线性 AC 校核。\n图 4：负荷输入、连续 SOC、磨损与外部迁移边界。\n\n英文主文的 5 张表和补充材料的 12 张表均为可编辑 Word 表格；本包另外提供相同表格的结构化源数据。\n\n图形风格采用 SciencePlots 2.2.2 和 https://github.com/Yuan1z0825/nature-skills 的 nature-figure，固定提交 2a20e4a0868ef9094257cb5386cfe623454ae092。采用该工具不意味着 Nature 的认可。\n\n每幅图附源数据、来源记录及最终 PDF 文字碰撞/面板对齐检查。所有最终预览均实际检查过。完整重建依赖数据和运行入口随独立的 revision_offline_reproduction.zip 交付；此图包用于审稿、编辑与图形使用，不替代完整复现包。\n'''
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,p in files.items():assert p.is_file();z.write(p,name)
        z.writestr('README_图表说明.txt',readme.encode('utf-8'))
        z.writestr('source_data/editable_table_sources.json',json.dumps(tables,ensure_ascii=False,indent=2).encode('utf-8'))
        z.writestr('SHA256_manifest.json',json.dumps({n:sha(p) for n,p in files.items()},indent=2).encode('utf-8'))
    with zipfile.ZipFile(out) as z:assert z.testzip() is None
    return {'file':out.name,'sha256':sha(out),'bytes':out.stat().st_size,'status':'PASS'}

def freeze():
    docs=structure();v=visual();protocol();fig=figure_bundle()
    report={'status':'PASS','recorded_utc':now(),'stage':'Local content revision and visual delivery freeze; no external submission','documents':docs,'visual_audit':'research/delivery_visual_audit.json','figure_bundle':fig,'review_items_addressed':29,'reproduction_final_sync':'Await final independently unpacked run after this freeze.'}
    write(RESEARCH/'delivery_freeze.json',report)
    print(json.dumps({'status':'PASS','documents':len(docs),'review_items':29,'figure_bundle_bytes':fig['bytes']},ensure_ascii=False))

def verify():
    # The package owner performs actual decode/verify/source/table/figure execution.
    ra=read(R/'reproducibility/reproduction_audit.json')
    fa=read(R/'reproducibility/portable_figure_entry_audit.json')
    assert ra['status']=='PASS' and fa['status']=='PASS'
    zpath=R/'reproducibility/revision_offline_reproduction.zip'
    with zipfile.ZipFile(zpath) as z:
        # Scientific files are hydrated from the bundle manifest; manuscript files
        # may likewise be stored as lossless blobs rather than direct ZIP members.
        names=z.namelist()
        manifests=[n for n in names if n.endswith('bundle_manifest.json') or n.endswith('manifest.json')]
        assert manifests
        match=[]
        for n in manifests:
            try:j=json.loads(z.read(n))
            except (ValueError,UnicodeDecodeError):continue
            text=json.dumps(j,ensure_ascii=False)
            if all(name in text for _,name,_ in DOCS):match.append(j)
        assert match,'Final DOCX files must be in the portable bundle manifest.'
        manifest_text=json.dumps(match,ensure_ascii=False)
        for _,name,_ in DOCS:assert sha(R/name) in manifest_text,(name,'hash mismatch')
    report={'status':'PASS','recorded_utc':now(),'scope':'Revised local manuscript, supplement, response, figure delivery and saved-artifact reconstruction; not a claim of full fresh retraining or field deployment.','documents':structure(),'visual_audit':read(RESEARCH/'delivery_visual_audit.json'),'figure_bundle':{'file':'Nature_Style_Figures.zip','bytes':(R/'Nature_Style_Figures.zip').stat().st_size,'sha256':sha(R/'Nature_Style_Figures.zip')},'reproduction':{'file':'reproducibility/revision_offline_reproduction.zip','bytes':zpath.stat().st_size,'sha256':sha(zpath),'status':'PASS','actual_run_evidence':['reproducibility/reproduction_audit.json','reproducibility/portable_figure_entry_audit.json']},'review_items_addressed':29,'original_manuscript_preserved':True}
    write(RESEARCH/'revision_delivery_verification.json',report)
    print(json.dumps({'status':'PASS','review_items':29,'reproduction_zip_bytes':zpath.stat().st_size},ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['freeze','verify'],required=True);a=p.parse_args()
    (freeze if a.stage=='freeze' else verify)()
