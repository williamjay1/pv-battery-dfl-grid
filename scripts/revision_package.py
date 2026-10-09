"""Rebuild a local, lossless, portable review bundle from completed artifacts.

Only writes revision_20261009/reproducibility. No raw data or remote publication.
NPZ files are deduplicated as exact ZIP member byte ranges, NOT rounded arrays.
Hydration concatenates those ranges and verifies the original whole-file SHA256.
"""
from pathlib import Path
import argparse, datetime, hashlib, importlib.metadata, json, os, re, shutil
import subprocess, sys, time, zipfile

ROOT = Path(__file__).resolve().parents[1]
REV = ROOT / 'revision_20261009'
OUT = REV / 'reproducibility'
TEMPLATES = OUT / 'templates'
ZIPNAME = 'revision_offline_reproduction.zip'

def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def sha_file(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def collect():
    """Explicit scientific inputs; excludes training caches and large renderings."""
    files={}; skipped=[]
    def add(p):
        if p.is_file():files[p.relative_to(ROOT).as_posix()]=p
    for pattern in ['scripts/*.py','checkpoints/*.pt','checkpoints/*.joblib',
                    'manuscript/references_verified.txt','datasets/*.json',
                    'datasets/ausgrid_source_clock_panel.npz',
                    'datasets/network_eulv_linearization.npz',
                    'datasets/network_eulv/*.txt','datasets/network_eulv/*.dss',
                    'results/*.json','results/analysis/*.json',
                    'research/*.json']:
        for p in ROOT.glob(pattern):add(p)
    for sub in ['models','cache','network_diagnostics','network_mechanism',
                'network_models','network_geography','research','manuscript','results']:
        for p in (REV/sub).rglob('*'):
            if p.suffix.lower() in ['.json','.npz','.pt','.joblib','.txt','.py','.csv','.dss']:
                add(p)
            elif sub=='manuscript' and p.suffix.lower()=='.docx':add(p)
    for p in REV.glob('*.json'):add(p)
    for name in ['EPSR_Revised_Manuscript.docx','EPSR_Supplementary_Material.docx','逐条修改说明.docx','Nature_Style_Figures.zip']:
        add(REV/name)
    for p in TEMPLATES.glob('*'):
        if p.suffix in ['.py','.md']:add(p)
    for p in (REV/'figures').glob('*'):
        if p.suffix in ['.json','.svg','.pdf']:add(p)
    external_audit=REV/'results/storenet_external_postrun_audit.json'
    external_completed=set()
    if external_audit.exists():
        external_report=json.loads(external_audit.read_text(encoding='utf-8'))
        if external_report.get('status')=='PASS':external_completed=set(external_report['checked_dispatch_files'])
    for p in (REV/'dispatch').glob('*.npz'):
        if 'pilot' in p.name:continue
        side=p.with_suffix('.json')
        if p.stem in external_completed:
            add(p)
            if side.exists():add(side)
            continue
        if not side.exists():skipped.append(p.name+': no completion sidecar');continue
        status=json.loads(side.read_text(encoding='utf-8'))
        if status.get('status') in ['running','failed'] or status.get('failed',0) or status.get('failures',[]):
            skipped.append(p.name+': incomplete/failed');continue
        add(p);add(side)
    vendor=REV/'vendor/Yuan1z0825-nature-skills-2a20e4a'
    for p in (vendor/'skills/nature-figure/scripts').glob('*.py'):add(p)
    add(vendor/'LICENSE')
    return files,skipped

def fragments(path):
    """Partition NPZ bytes at ZIP local-header boundaries; preserves metadata."""
    with zipfile.ZipFile(path) as z:
        cuts=sorted(set([0]+[i.header_offset for i in z.infolist()]+[z.start_dir,path.stat().st_size]))
    with path.open('rb') as f:
        for a,b in zip(cuts,cuts[1:]):
            f.seek(a);data=f.read(b-a)
            yield sha_bytes(data),data

def estimate(files):
    seen=set();logical=unique=0; npz_count=0
    for name,p in files.items():
        logical+=p.stat().st_size
        if p.suffix=='.npz':
            npz_count+=1
            for digest,b in fragments(p):
                if digest not in seen:unique+=len(b);seen.add(digest)
        else:unique+=p.stat().st_size
    return {'logical_input_bytes':logical,'estimated_payload_bytes_before_text_compression':unique,
            'saved_duplicate_bytes':logical-unique,'npz_files':npz_count,
            'unique_npz_fragments':len(seen),'d_free_bytes':shutil.disk_usage(OUT).free}

def portable_copy(name,data):
    """Only non-scientific path portability changes; originals stay untouched."""
    old=data
    if name=='scripts/revision_figures.py':
        data=data.replace(b"sys.path.insert(0,r'C:\\Users\\Administrator\\.agents\\skills\\sci-figures')",
                          b"sys.path.insert(0,str(ROOT/'portable_vendor'))")
    if name=='scripts/revision_package.py':
        data=data.replace(b"Path(r'C:\\Users\\Administrator\\.agents\\skills\\sci-figures\\figstyle.py')",
                          b"ROOT/'portable_vendor/figstyle.py'")
    if name=='scripts/revision_initialize.py':
        data=data.replace(b"ROOT = Path('D:/MLWork/pv_battery_dfl_grid_20261005')",b"ROOT = Path(__file__).resolve().parents[1]")
    if name=='scripts/prepare_ausgrid.py':
        data=data.replace(b"if out.drive.upper()!='D:':\n    raise ValueError('Processed research outputs must be on D: under the current storage policy')\n",b'')
    if name=='scripts/network_verify.py':
        data=data.replace(b'RAW = Path("F:/AcademicData/pv_battery_dfl_grid_20261005/raw/network_20261005")',
                          b'RAW = Path(__file__).resolve().parents[1]/"raw_inputs"/"network_20261005"')
        data=data.replace(b'shutil.disk_usage("F:/")',b'shutil.disk_usage(RAW.parent)')
    if name=='revision_20261009/research/prepare_storenet_external.py':
        data=data.replace(b"sys.path.insert(0, r'D:\\MLWork\\pv_battery_dfl_grid_20261005\\temp\\pydeps')\n",b'')
        data=data.replace(b"ROOT = Path(r'D:\\MLWork\\pv_battery_dfl_grid_20261005\\revision_20261009')",b'ROOT = Path(__file__).resolve().parents[1]')
        data=data.replace(b"path = Path(record['path'])",b"path = ROOT.parent / 'raw_inputs' / 'storenet' / record['path'].replace('\\\\','/').split('/')[-1]")
    return data,old!=data

def environment():
    names=['numpy','pandas','scipy','torch','scikit-learn','joblib','osqp','cvxpy','cvxpylayers','diffcp','clarabel','matplotlib','SciencePlots','Pillow','opendssdirect.py','simbench','pandapower']
    versions={}
    for n in names:
        try:versions[n]=importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError:versions[n]=None
    return {'python':sys.version,'packages':versions,
            'note':'Recorded build environment; minimal verification only needs Python + NumPy. Dependencies are not bundled.'}

def build(files,skipped,est):
    if shutil.disk_usage(OUT).free < est['logical_input_bytes']*2+est['estimated_payload_bytes_before_text_compression']:
        raise RuntimeError('Insufficient workspace space for archive and independent extraction check')
    target=OUT/ZIPNAME;temp=OUT/(ZIPNAME+'.partial')
    manifest={'created_utc':now(),'status':'rebuildable working snapshot, not a publication freeze',
      'npz_encoding':'Exact local ZIP-record byte ranges deduplicated by SHA256; original NPZ hash restored.',
      'files':{},'omitted_incomplete_dispatches':skipped,'portability_changes':[],'estimate':est}
    seen=set()
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for name,p in sorted(files.items()):
            before=p.stat();origsha=sha_file(p)
            if p.suffix=='.npz':
                parts=[]
                for digest,b in fragments(p):
                    parts.append({'sha256':digest,'bytes':len(b)})
                    if digest not in seen:
                        z.writestr('objects/'+digest,b,compress_type=zipfile.ZIP_STORED);seen.add(digest)
                entry={'encoding':'npz_fragments','bytes':before.st_size,'sha256':origsha,'parts':parts}
            else:
                data,changed=portable_copy(name,p.read_bytes())
                z.writestr(name,data)
                entry={'encoding':'file','bytes':len(data),'sha256':sha_bytes(data),'source_sha256':origsha}
                if changed:manifest['portability_changes'].append(name)
            after=p.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                raise RuntimeError('Source changed while reading; rerun after completion: '+name)
            manifest['files'][name]=entry
        extras={'reproduce.py':TEMPLATES/'reproduce.py','README.md':TEMPLATES/'README.md',
                'DATA_SOURCES.md':TEMPLATES/'DATA_SOURCES.md',
                'portable_vendor/figstyle.py':Path(r'C:\Users\Administrator\.agents\skills\sci-figures\figstyle.py')}
        for name,p in extras.items():
            data=p.read_bytes();z.writestr(name,data)
            manifest['files'][name]={'encoding':'file','bytes':len(data),'sha256':sha_bytes(data)}
        env=environment();z.writestr('environment.json',json.dumps(env,indent=2))
        z.writestr('requirements-minimal.txt','numpy\n')
        z.writestr('requirements-figures.txt','numpy\nmatplotlib\nSciencePlots\nPillow\n')
        z.writestr('THIRD_PARTY_NOTICES.txt','Figure alignment auditor: Yuan1z0825/nature-skills commit 2a20e4a0868ef9094257cb5386cfe623454ae092, Apache-2.0; license preserved in revision_20261009/vendor.\nfigstyle.py is a local scientific-style helper copied for this private review bundle; its source and only import-path adaptation are recorded. No claim of a blanket license over third-party data or code.\nEPRI network license is preserved under datasets/network_eulv/EPRI_License.txt. Data attribution is in DATA_SOURCES.md.\n')
        z.writestr('bundle_manifest.json',json.dumps(manifest,indent=2))
    os.replace(temp,target)
    result={'created_utc':now(),'zip':target.name,'sha256':sha_file(target),'zip_bytes':target.stat().st_size,
      'files':len(manifest['files']),'npz_files':sum(v['encoding']=='npz_fragments' for v in manifest['files'].values()),
      'dispatch_npz_files':sum(k.startswith('revision_20261009/dispatch/') and k.endswith('.npz') for k in manifest['files']),
      'portability_changes':manifest['portability_changes'],'estimate':est,'skipped_dispatches':skipped,
      'status':'built; independent extraction verification not yet run'}
    (OUT/'package_build_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)
    return target,result

def smoke(archive,build_audit,include_figures=False):
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    test=OUT/('extraction_check_'+stamp);test.mkdir(exist_ok=False)
    with zipfile.ZipFile(archive) as z:z.extractall(test)
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONUTF8']='1'
    checks=[]
    # All actions run from the extracted folder, never against project inputs.
    commands=[['verify'],['sources'],['tables']]+([['figures']] if include_figures else [])
    for args in commands:
        cmd=[sys.executable,'-B',str(test/'reproduce.py'),*args]
        started=time.perf_counter();proc=subprocess.run(cmd,cwd=test,env=env,capture_output=True,text=True,encoding='utf-8')
        log='smoke_'+args[0]+'.log';(test/log).write_text(proc.stdout+'\nSTDERR\n'+proc.stderr,encoding='utf-8')
        row={'command':['python','reproduce.py',*args],'returncode':proc.returncode,
             'seconds':time.perf_counter()-started,'log':str((test/log).relative_to(OUT))}
        checks.append(row);print(json.dumps(row),flush=True)
        if proc.returncode:break
    result=dict(build_audit,status='PASS' if all(c['returncode']==0 for c in checks) and len(checks)==len(commands) else 'FAIL',
      extraction_directory=str(test.relative_to(OUT)),checks=checks,
      scope='Executed costs/state/paired means, figure-source dependencies and table regeneration from an independently extracted package. Figure rendering executed only if listed in checks. Not fresh training or AC rerun. No dependencies downloaded.',
      checked_utc=now())
    (OUT/'reproduction_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    # The one deliverable ZIP contains the executed audit, but not another data copy.
    with zipfile.ZipFile(archive,'a',compression=zipfile.ZIP_DEFLATED) as z:
        embedded={k:v for k,v in result.items() if k not in ['sha256','zip_bytes']}
        embedded['archive_identity_note']='Final ZIP hash and byte length are recorded in the adjacent reproduction_audit.json after this audit is appended; no circular self-hash is asserted.'
        z.writestr('EXECUTED_REPRODUCTION_AUDIT.json',json.dumps(embedded,indent=2))
        for name in ['verification_audit.json','source_check_audit.json','table_rebuild_audit.json','figure_rebuild_audit.json']:
            p=test/'reproduced'/name
            if p.exists():z.write(p,'executed_checks/'+name)
    result['sha256']=sha_file(archive);result['zip_bytes']=archive.stat().st_size
    (OUT/'reproduction_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    if result['status']!='PASS':raise RuntimeError('Extraction test failed; see reproduction_audit.json and smoke logs')
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--estimate',action='store_true');ap.add_argument('--smoke',action='store_true');ap.add_argument('--smoke-figures',action='store_true');a=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);files,skipped=collect();est=estimate(files)
    (OUT/'package_size_estimate.json').write_text(json.dumps(est,indent=2),encoding='utf-8');print(json.dumps(est),flush=True)
    if a.estimate:return
    target,audit=build(files,skipped,est)
    if a.smoke or a.smoke_figures:print(json.dumps(smoke(target,audit,a.smoke_figures)),flush=True)

if __name__=='__main__':main()
