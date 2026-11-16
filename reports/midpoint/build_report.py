#!/usr/bin/env python3
"""Build an editable checkpoint report from a committed, verified evidence snapshot.

Requires python-docx. PDF export and visual review are a separate documented step.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from string import Template
import subprocess
from urllib.parse import quote

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def table_rows(rows):
    return '\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)


def hyperlink(paragraph, label, target):
    rel = paragraph.part.relate_to(target,
        'http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink',is_external=True)
    element=OxmlElement('w:hyperlink');element.set(qn('r:id'),rel)
    run=OxmlElement('w:r'); props=OxmlElement('w:rPr')
    color=OxmlElement('w:color');color.set(qn('w:val'),'000000');props.append(color)
    under=OxmlElement('w:u');under.set(qn('w:val'),'single');props.append(under)
    run.append(props);text=OxmlElement('w:t');text.text=label;run.append(text)
    element.append(run);paragraph._p.append(element)


def inline(paragraph, text):
    pattern=r'\[([^\]]+)\]\(([^)]+)\)|\*\*([^*]+)\*\*|`([^`]+)`'
    position=0
    for match in re.finditer(pattern,text):
        paragraph.add_run(text[position:match.start()])
        if match.group(1):hyperlink(paragraph,match.group(1),match.group(2))
        elif match.group(3):paragraph.add_run(match.group(3)).bold=True
        else:
            run=paragraph.add_run(match.group(4));run.font.name='Courier New';run.font.size=Pt(9)
        position=match.end()
    paragraph.add_run(text[position:])


def make_docx(markdown, path):
    doc=Document();section=doc.sections[0]
    section.page_width=Inches(8.5);section.page_height=Inches(11)
    section.top_margin=Inches(.65);section.bottom_margin=Inches(.65)
    section.left_margin=Inches(.7);section.right_margin=Inches(.7)
    section.footer_distance=Inches(.3)
    normal=doc.styles['Normal'];normal.font.name='Arial';normal.font.size=Pt(10)
    normal.paragraph_format.space_after=Pt(6);normal.paragraph_format.line_spacing=1.08
    for name,size in [('Title',18),('Heading 1',14),('Heading 2',11)]:
        style=doc.styles[name];style.font.name='Arial';style.font.size=Pt(size)
        style.font.color.rgb=RGBColor(0,0,0);style.font.bold=True
        style.paragraph_format.space_before=Pt(9 if name!='Title' else 0)
        style.paragraph_format.space_after=Pt(6);style.paragraph_format.keep_with_next=True
    for border in list(doc.styles.element.iter(qn('w:pBdr'))):
        border.getparent().remove(border)
    footer=section.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
    field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');footer._p.append(field)
    for run in footer.runs:run.font.size=Pt(8)
    lines=markdown.splitlines();i=0;break_next=False
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line=='<!-- pagebreak -->':break_next=True;i+=1;continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].startswith('|'):
                cells=[x.strip() for x in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?',x) for x in cells):rows.append(cells)
                i+=1
            n=len(rows[0]);table=doc.add_table(rows=0,cols=n);table.autofit=False
            # Narrative columns need more room; compact numeric comparisons stay legible.
            header=rows[0]
            if header[0]=='Team member':widths=[1.25,.65,5.2]
            elif header[0]=='Frozen split':widths=[1.4,.8,4.9]
            elif header[0]=='Epoch':widths=[.55,1.4,1.45,1.85,1.85]
            elif header[0]=='Step':widths=[1.0,1.3,2.6,2.2]
            elif n==2:widths=[4.6,2.5]
            elif n==3:widths=[3.5,1.8,1.8]
            elif n==6:widths=[2.4,.55,1.0,.95,1.15,1.05]
            else:widths=[7.1/n]*n
            for col,width in zip(table.columns,widths):col.width=Inches(width)
            borders=OxmlElement('w:tblBorders')
            for edge in ('top','left','bottom','right','insideH','insideV'):
                b=OxmlElement('w:'+edge);b.set(qn('w:val'),'single');b.set(qn('w:sz'),'4');b.set(qn('w:color'),'D9D9D9');borders.append(b)
            table._tbl.tblPr.append(borders)
            for row_index,values in enumerate(rows):
                row=table.add_row();pr=row._tr.get_or_add_trPr()
                no_split=OxmlElement('w:cantSplit');pr.append(no_split)
                if row_index==0:
                    repeat=OxmlElement('w:tblHeader');pr.append(repeat)
                for j,(cell,value) in enumerate(zip(row.cells,values)):
                    cell.width=Inches(widths[j]);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
                    tcpr=cell._tc.get_or_add_tcPr();margins=OxmlElement('w:tcMar')
                    for edge in ('top','left','bottom','right'):
                        m=OxmlElement('w:'+edge);m.set(qn('w:w'),'75');m.set(qn('w:type'),'dxa');margins.append(m)
                    tcpr.append(margins)
                    if row_index==0:
                        shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'EEEEEE');tcpr.append(shade)
                    p=cell.paragraphs[0];p.paragraph_format.space_after=Pt(0);p.paragraph_format.line_spacing=1.03
                    p.alignment=WD_ALIGN_PARAGRAPH.LEFT if j==0 or len(value)>35 else WD_ALIGN_PARAGRAPH.CENTER
                    inline(p,value)
                    for r in p.runs:r.font.size=Pt(9);r.bold=row_index==0
            doc.add_paragraph().paragraph_format.space_after=Pt(1)
            continue
        if line.startswith('# '):p=doc.add_paragraph(line[2:],style='Title')
        elif line.startswith('## '):p=doc.add_paragraph(line[3:],style='Heading 1')
        elif line.startswith('### '):p=doc.add_paragraph(line[4:],style='Heading 2')
        else:
            p=doc.add_paragraph();inline(p,line)
        p.paragraph_format.page_break_before=break_next;break_next=False
        p.paragraph_format.widow_control=True;i+=1
    doc.core_properties.title='Brain MRI VLM Midpoint Report'
    doc.core_properties.subject='COMS 6998 HPML Team 32 checkpoint results'
    doc.core_properties.author='Owen Shi; Chengshun Tan; Ai Hsiao; Trung Kien Giang'
    doc.core_properties.keywords='SmolVLM; MRI; LoRA; inference; profiling; midpoint'
    path.parent.mkdir(parents=True,exist_ok=True);doc.save(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-commit',required=True)
    args=parser.parse_args()
    commit=subprocess.check_output(['git','rev-parse',args.evidence_commit],cwd=ROOT,text=True).strip()
    hashes={}
    def source(name):
        raw=subprocess.check_output(['git','show',f'{commit}:{name}'],cwd=ROOT)
        hashes[name]=hashlib.sha256(raw).hexdigest();return json.loads(raw)
    s5=source('experiments/step5/summary.json');s6=source('experiments/step6/summary.json')
    s8=source('experiments/step8/summary.json')
    for step in (3,4,7):source(f'experiments/step{step}/summary.json')
    if s8['status']!='verified-complete' or s8['wandb_verification']['status']!='passed':
        raise ValueError('The report requires completed Step 8 execution and artifact verification')
    b=s8['timing']['baseline'];v=s8['timing']['image1024'];q=s8['quality'];c=s8['comparison']
    hardware=source(f"experiments/step8/runs/{s8['gpu_job']['JobIDRaw']}/cpu-hardware.json")
    cpu_model=next(item['data'] for item in hardware['lscpu'] if item['field']=='Model name:')
    sweep=[]
    for batch in [1,2,4,8,16,32]:
        x=source(f'experiments/step6/runs/13932242/sweep-b{batch:03}/summary.json')
        sweep.append([batch,f"{x['mean_batch_ms']:.3f}",f"{x['p95_batch_ms']:.3f}",f"{x['throughput_images_s']:.6f}",f"{x['peak_allocated_bytes']/2**30:.3f}"])
    base=f'https://github.com/oowenn/brain-mri-vlm/blob/{commit}/'
    if c['preliminary_success']:
        opening=f"The paired run reduced mean batch latency by {100*c['latency_reduction_fraction']:.2f}% ({c['speedup']:.2f}x speedup) while passing the validation quality gate."
        interpretation=f"The variant was {c['speedup']:.3f}x faster by mean batch time and reduced peak allocated memory by {100*c['peak_allocated_reduction_fraction']:.2f}%. Its macro-F1 changed by {c['macro_f1_change']:+.6f}; the predefined validation gate passed. This closes the first-optimization evidence gap with a measured preliminary improvement."
        next_work='retain the 1024 image budget as a validation-qualified candidate and confirm its benefit in another paired allocation before combining it with another technique.'
    else:
        opening=f"The paired run measured {c['speedup']:.2f}x baseline-to-variant latency ratio, but the variant did not meet all success criteria. We retain it as a documented preliminary experiment."
        interpretation=f"The baseline-to-variant mean latency ratio was {c['speedup']:.3f}x and peak allocated memory changed by {-100*c['peak_allocated_reduction_fraction']:+.2f}%. Macro-F1 changed by {c['macro_f1_change']:+.6f}. The quality gate {'passed' if c['quality_gate_passed'] else 'failed'}; {c['interpretation']}. This closes the evidence gap with an actual before/after result, without claiming an accepted optimization."
        next_work='keep the 2048 baseline as the accepted reference while investigating the measured 1024 tradeoff; do not tune against test results.'
    def distribution(d):return ', '.join(f'{n} tokens for {count} images' for n,count in sorted(d.items(),key=lambda x:int(x[0])))
    stage_b=s8['stages']['baseline']['stages'];stage_v=s8['stages']['image1024']['stages']
    stage_text='; '.join(f"{label} {stage_b[key]['mean_ms']:.3f} to {stage_v[key]['mean_ms']:.3f} ms" for key,label in [('preprocessing','preprocessing'),('transfer','transfer'),('vision','vision'),('prefill','prefill'),('decode','decode')])
    values=dict(optimization_opening=opening,optimization_interpretation=interpretation,next_image_work=next_work,
        epoch_rows=table_rows([[e['epoch'],f"{e['training_loss']:.6f}",f"{e['validation_loss']:.6f}",f"{100*e['metrics']['accuracy']:.2f}%",f"{e['metrics']['macro_f1']:.6f}"] for e in s5['epochs']]),
        sweep_rows=table_rows(sweep),step8_job=s8['gpu_job']['JobIDRaw'],step8_node=s8['hostname'],step8_run=s8['wandb_run_id'],cpu_model=cpu_model,
        step8_publisher=s8['publisher_job']['JobIDRaw'],step8_files=s8['wandb_verification']['downloaded_files_verified'],
        step8_outcome=f"Completed; {s8['unit_tests_passed']} tests; {s8['gpu_job']['Elapsed']}; quality gate {'passed' if c['quality_gate_passed'] else 'failed'}",
        optimization_rows=table_rows([
            ['Mean batch latency ms',f"{b['mean_batch_ms']:.3f}",f"{v['mean_batch_ms']:.3f}"],
            ['Median batch latency ms',f"{b['p50_batch_ms']:.3f}",f"{v['p50_batch_ms']:.3f}"],
            ['p95 batch latency ms',f"{b['p95_batch_ms']:.3f}",f"{v['p95_batch_ms']:.3f}"],
            ['Images per second',f"{b['throughput_images_s']:.6f}",f"{v['throughput_images_s']:.6f}"],
            ['Peak allocated GPU GiB',f"{b['peak_allocated_bytes']/2**30:.3f}",f"{v['peak_allocated_bytes']/2**30:.3f}"],
            ['Peak reserved GPU GiB',f"{b['peak_reserved_bytes']/2**30:.3f}",f"{v['peak_reserved_bytes']/2**30:.3f}"],
            ['Validation accuracy',f"{100*q['baseline']['accuracy']:.2f}%",f"{100*q['image1024']['accuracy']:.2f}%"],
            ['Validation macro-F1',f"{q['baseline']['macro_f1']:.6f}",f"{q['image1024']['macro_f1']:.6f}"],
            ['Invalid responses',q['baseline']['invalid_count'],q['image1024']['invalid_count']]]),
        recall_rows=table_rows([[label,f"{100*q['baseline']['per_class'][label]['recall']:.2f}%",f"{100*q['image1024']['per_class'][label]['recall']:.2f}%"] for label in ['glioma','meningioma','notumor','pituitary']]),
        baseline_shape=str(s8['context']['baseline']['stage_pixel_shapes']),variant_shape=str(s8['context']['image1024']['stage_pixel_shapes']),
        baseline_context=distribution(s8['context']['baseline']['validation_input_token_counts']),
        variant_context=distribution(s8['context']['image1024']['validation_input_token_counts']),
        variant_output_range='–'.join(map(str,s8['context']['image1024']['generated_token_range'])),
        optimization_stage_explanation='Diagnostic mean stage changes were: '+stage_text+'. The largest absolute savings came from preprocessing and vision work; cached decoding changed little.',
        paired_speedups=', '.join(f"{t['speedup']:.3f}x" for t in s8['paired_trials']),
        wandb='https://forge.coreweave.com/wandb/coms6998-team32-fall2026/brain-mri-vlm/runs',evidence_commit=commit,
        e0=base+quote('COMS 6998 Proposal (5).pdf'),e1=base+'experiments/data_finalization.md',e2a=base+'experiments/step3/README.md',
        **{'e'+str(n-2):base+f'experiments/step{n}/README.md' for n in [4,5,6,7,8]})
    markdown=Template((HERE/'report_template.md').read_text()).substitute(values)
    (HERE/'report.md').write_text(markdown)
    path=ROOT/'output/docx/brain-mri-vlm-midpoint-report.docx';make_docx(markdown,path)
    record=dict(evidence_commit=commit,source_sha256=hashes,generated_utc=datetime.now(timezone.utc).isoformat(),
                docx=str(path.relative_to(ROOT)),docx_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                markdown_sha256=hashlib.sha256(markdown.encode()).hexdigest(),step8_job=s8['gpu_job']['JobIDRaw'])
    (HERE/'build-record.json').write_text(json.dumps(record,indent=2)+'\n')
    print(path)


if __name__=='__main__':main()
