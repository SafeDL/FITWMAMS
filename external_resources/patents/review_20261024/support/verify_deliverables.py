from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
import re,json,hashlib,subprocess,sys
sys.path.insert(0,'/tmp/patent_drafting_deps')
from docx import Document
from PIL import Image,ImageDraw

OUT=Path(__file__).resolve().parents[1]
SOURCE=OUT.parent/'revision_20261023/技术交底书_自然驾驶事件约束的自动驾驶多失效域测试方法_修订稿.docx'
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
W='{'+NS['w']+'}'
manifest=json.loads((OUT/'support/source_manifest.json').read_text())
for entry in manifest['sources'][:2]:
    assert hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest()==entry['sha256']
results={}
for p in OUT.glob('*.docx'):
    ismain='交底书' in p.name
    with ZipFile(p) as z:
        assert z.testzip() is None
        for name in z.namelist():
            if name.endswith('.xml') or name.endswith('.rels'):E.fromstring(z.read(name))
        root=E.fromstring(z.read('word/document.xml'))
        settings=E.fromstring(z.read('word/settings.xml'))
        assert settings.xpath('string(//w:compatSetting[@w:name="compatibilityMode"]/@w:val)',namespaces=NS)=='15'
        wt=''.join(root.xpath('//w:t/text()',namespaces=NS))
        mt=root.xpath('//m:t/text()',namespaces=NS)
        assert not any(re.search('[\u4e00-\u9fff，。；：]',v) for v in mt)
        assert '$' not in wt and not re.search(r'\\(?:frac|sum|sqrt|theta|omega|pi)',wt)
        assert not root.xpath('//w:pict',namespaces=NS)
        assert not [n for n in z.namelist() if 'equationxml' in n.lower()]
        assert not any(re.search(r'\b(?:EQ|EMBED)\b',v) for v in root.xpath('//w:instrText/text()',namespaces=NS))
        assert not root.xpath('//w:ins | //w:del',namespaces=NS)
        rels=E.fromstring(z.read('word/_rels/document.xml.rels'))
        rd={r.get('Id'):r for r in rels}
        for rid in root.xpath('//w:hyperlink/@r:id',namespaces=NS):
            assert rid in rd and rd[rid].get('Target','').startswith('https://')
        if ismain:
            assert wt.count('【附图占位：待绘制】')==8
            assert '主要公开参考文献' not in wt and '9 替代实施方式' not in wt and '10 附图占位' not in wt
            assert '12个后续名额' not in wt
            # The requested section 6 has no tables.
            active=False
            for el in root.find(W+'body'):
                text=''.join(el.xpath('.//w:t/text()',namespaces=NS))
                if text=='6 关键技术特征与欲保护内容':active=True
                if text=='7 有益效果':active=False
                if active:assert el.tag!=W+'tbl'
            labels=[]
            for para in root.xpath('//w:p',namespaces=NS):
                text=''.join(para.xpath('.//w:t/text()',namespaces=NS))
                m=re.fullmatch(r'（(\d+)）',text)
                if m:
                    labels.append(int(m.group(1)))
                    assert para.xpath('string(w:pPr/w:jc/@w:val)',namespaces=NS)=='right'
            assert labels==list(range(1,15))
            for tag in ['commentRangeStart','commentRangeEnd','commentReference']:
                ids=root.xpath('//w:'+tag+'/@w:id',namespaces=NS)
                assert sorted(ids)==list('01234567')
            with ZipFile(SOURCE) as sz:
                commentparts=[n for n in sz.namelist() if 'comments' in n.lower() and n.endswith('.xml')]
                for n in commentparts:assert sz.read(n)==z.read(n)
            cr=E.fromstring(z.read('word/comments.xml'))
            assert len(cr)==8
            assert any(r.get('Type','').endswith('/comments') for r in rels)
        assert Document(p)
        pdf=p.with_suffix('.pdf');assert pdf.exists()
        text=subprocess.check_output(['pdftotext','-layout',str(pdf),'-'],text=True)
        assert not re.search('[¿❑�]',text)
        info=subprocess.check_output(['pdfinfo',str(pdf)],text=True)
        pages=int(re.search(r'Pages:\s+(\d+)',info).group(1))
        assert len([v for v in text.split('\f') if len(v.strip())>30])==pages
        result={'zip_xml_valid':True,'docx_loader_reopened':True,'pdf_pages':pages,'math_objects':len(root.xpath('//m:oMath',namespaces=NS)),'native_math_paragraphs':len(root.xpath('//m:oMathPara',namespaces=NS)),'text_characters':len(wt),'image_formula_remnants':0,'comments':8 if ismain else 0,'tracked_insertions_deletions':0,'hyperlinks_verified':len(root.xpath('//w:hyperlink',namespaces=NS)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        if ismain:
            result.update(numbered_equations=14,original_comment_parts_byte_preserved=True,figure_placeholders=8,section6_tables=0)
            bbox=OUT/'support/main_bbox.html'
            subprocess.check_call(['pdftotext','-bbox',str(pdf),str(bbox)])
            br=E.parse(str(bbox));bns={'h':'http://www.w3.org/1999/xhtml'}
            words=br.findall('.//h:word',bns)
            numbers=[v for v in words if re.fullmatch(r'（\d+）',v.text or '')]
            assert len(numbers)==14
            right=[float(v.get('xMax')) for v in numbers]
            assert max(right)-min(right)<.2
            result['pdf_equation_number_right_edge_pt']=right[0]
        results[p.name]=result
        label='main' if ismain else 'analysis'
        (OUT/'support'/f'{label}_pdf.txt').write_text(text)
        (OUT/'support'/f'{label}_current.txt').write_text('\n'.join(''.join(x.xpath('.//w:t/text() | .//m:t/text()',namespaces=NS)) for x in root.xpath('//w:body//w:p',namespaces=NS)))
for label in ['main','analysis']:
    files=sorted((OUT/'support').glob(label+'-*.png'));thumbs=[]
    for f in files:
        im=Image.open(f).convert('RGB');im.thumbnail((360,510));t=Image.new('RGB',(380,542),'#ddd');t.paste(im,((380-im.width)//2,22));ImageDraw.Draw(t).text((8,4),f.stem,fill='black');thumbs.append(t)
    contact=Image.new('RGB',(4*380,((len(thumbs)+3)//4)*542),'white')
    for i,im in enumerate(thumbs):contact.paste(im,((i%4)*380,(i//4)*542))
    contact.save(OUT/'support'/f'{label}_contact_sheet.png')
for p in (OUT/'support').glob('check_math-*.png'):p.unlink()
report={'date':'2026-10-24','checks_passed':True,'documents':results,'original_source_documents_unchanged':True,'experiments_or_training_started':False,'review_response_map_entries':len(json.loads((OUT/'support/review_response_map.json').read_text())),'native_child_replies_added':False,'comment_delivery':'保留原始批注及元数据，在独立说明中逐条答复；未要求原生回复框','office_validation':'LibreOffice 7.3加载及PDF导出；Microsoft Word桌面／连接会话不可用，未进行Word保存关闭重开','layout_review':'全页缩略图及放大公式页面检查；编号右边缘一致，公式及符号说明保持同页'}
(OUT/'support/qa_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False,indent=2))
