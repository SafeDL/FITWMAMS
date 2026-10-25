"""Validate rendered artifacts and their identity against existing experiment data."""
from pathlib import Path
import csv,json,re
import numpy as np
from PIL import Image,ImageDraw

OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parents[2]
manifest=json.loads((OUT/'manifest.json').read_text())
assert len(manifest['figures'])==11
assert {x['requirement'] for x in manifest['skipped']}=={'E10','E12','E13','E15','E16'}
assert not manifest['model_executions_started'] and not manifest['training_started']
log=(OUT/'_support/render.log').read_text()
assert not re.search(r'Glyph|findfont|UserWarning|Traceback',log)
reports=[]
for f in manifest['figures']:
    stem=f['stem'];p=OUT/(stem+'.png')
    im=Image.open(p);assert min(im.size)>850
    dpi=im.info['dpi'];assert min(dpi)>299
    im.verify()
    reports.append({'stem':stem,'png_size':im.size,'dpi':dpi,'png_integrity_verified':True})

def csvarray(name):return np.genfromtxt(OUT/'data'/name,delimiter=',',names=True,encoding='utf-8')
f=next(f for f in manifest['figures'] if f['stem'].startswith('E01'))
a=csvarray('E01_natural_event.csv')
with np.load(ROOT/'results/evt/natural_risk_traces.npz') as z:
    np.testing.assert_array_equal(a['prefix_event_risk'].astype(np.float32),z['risk_trace'][f['row']])
with np.load(ROOT/'results/hierarchical_world_model/bridge/intervention_traces.npz') as z:
    for stem,n,j in [('E05_braking_response',1,1),('E06_leader_release',3,1),('E07_lateral_response',3,5)]:
        a=csvarray(stem+'.csv')
        np.testing.assert_allclose(a['takeover_ax_mps2'],z[f'{n}_takeover_20261015/actions'][:,j,0],rtol=0,atol=0)
        np.testing.assert_allclose(a['tracker_ax_mps2'],z[f'{n}_tracker_20261015/actions'][:,j,0],rtol=0,atol=0)
    a=csvarray('E09_handover_timing.csv')
    np.testing.assert_array_equal(a['owned'],z['1_takeover_20261015/takeover_owned'][:,1])
    np.testing.assert_array_equal(a['realised_motion_changed'],z['1_takeover_20261015/takeover_motion_changed'][:,1])
a=csvarray('E03_natural_replay.csv');assert len(a)==9841
assert len(np.unique(a['row']))==9841

# Compact internal review sheets, not publication figure content.
files=[OUT/(f['stem']+'.png') for f in manifest['figures']]
for part,chunk in enumerate([files[:6],files[6:]],1):
    thumbs=[]
    for p in chunk:
        im=Image.open(p).convert('RGB');im.thumbnail((740,460));tile=Image.new('RGB',(760,490),'white');tile.paste(im,((760-im.width)//2,25));ImageDraw.Draw(tile).text((12,6),p.stem,fill='black');thumbs.append(tile)
    sheet=Image.new('RGB',(1520,490*((len(thumbs)+1)//2)),'#e8e8e8')
    for i,t in enumerate(thumbs):sheet.paste(t,((i%2)*760,(i//2)*490))
    sheet.save(OUT/'_support'/f'contact_{part}.png')

assert manifest['formats']==['png_300dpi']
assert not list(OUT.glob('*.pdf')) and not list(OUT.glob('*.svg'))
readme=(OUT/'README.md').read_text()
for link in re.findall(r'\[[^\]]+\]\(([^)]+)\)',readme):assert (OUT/link).exists(),link
report={'checks_passed':True,'date':'2026-10-25','formats':['png_300dpi'],'figures':reports,'skipped':[x['requirement'] for x in manifest['skipped']],'data_checks':'Source risk array, paired executed controls, timing flags and scene-level statistical unit verified.','font_warnings':0,'models_or_simulations_executed':False,'visual_review':'Existing PNGs preserved; previously reviewed all-figure contact sheets and enlarged snapshots, timing markers, Chinese labels, legends and residual panels.'}
(OUT/'_support/qa_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({'checks_passed':True,'png_figures':len(reports),'skipped':report['skipped']},ensure_ascii=False))
