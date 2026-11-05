#!/usr/bin/env python3
"""DM3D v5: PDF/source-conditioned, mathematically bounded recursive 3D search.

Builds on dm3d_septillion_inward_v4.py (packaged alongside this file).
Pure 3D XYZ operations, a genuine 2-scale contracting state update,
verified codec candidates, a Pareto archive, and PDF-to-PDF source-conditioned
feedback. Astronomical cardinality is symbolic only, NEVER enumerated.
"""
from __future__ import annotations
import argparse, hashlib, io, json, math, random, statistics, time, zipfile, zlib
from pathlib import Path
import numpy as np
import fitz
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
import dm3d_septillion_inward_v4 as core

VERSION='5.0.0'
WHITE=HexColor('#f1fbff'); LIGHT=HexColor('#abcad6'); BLUE=HexColor('#35c9f4')
GREEN=HexColor('#63eac5'); GOLD=HexColor('#ffd18c')
BG=HexColor('#08131f'); PANEL=HexColor('#11263a'); GRID=HexColor('#294a60')
GENE_SHAPE=core.GENE_SHAPE


def hash_data(data): return hashlib.sha256(data).hexdigest()
def canonical(x): return json.dumps(x,sort_keys=True,separators=(',',':')).encode()


def neighborhood(f):
    return sum((np.roll(f,k,axis) for axis in range(3) for k in (-1,1)))/6


def multiscale_field(state,target,steps=5):
    """Multiresolution contraction. Operator has Lipschitz constant <= 0.72.
    F'=0.55F + 0.10 mean6(F) +0.07 expand(mean2x2x2(F)) +0.28 target.
    All components are non-expansive in ||.||_inf; 0.55+0.10+0.07=0.72.
    """
    f=np.asarray(state,dtype=np.float32).copy()
    y=np.asarray(target,dtype=np.float32)
    if f.shape!=y.shape or f.ndim!=3 or f.shape[0]%2 or not (f.shape[0]==f.shape[1]==f.shape[2]):
        raise ValueError('even XYZ cubes of equal shape required')
    changes=[]
    for _ in range(steps):
        s=f.shape[0]//2
        coarse=f.reshape(s,2,s,2,s,2).mean(axis=(1,3,5)).astype(np.float32)
        lifted=coarse.repeat(2,0).repeat(2,1).repeat(2,2)
        nf=.55*f+.10*neighborhood(f)+.07*lifted+.28*y
        changes.append(float(np.max(np.abs(nf-f))))
        f=nf.astype(np.float32)
        if not np.isfinite(f).all(): raise ArithmeticError('non-finite field')
    return f, changes


def same_genome(g): return tuple(int(v) for v in g)


def quality_gate(r, threshold):
    return r['psnr_db'] is None or r['psnr_db']>=threshold


def candidate_score(r,baseline_ms,psnr_floor):
    if not quality_gate(r,psnr_floor):return None
    # Favor shorter encodings; penalize PSNR reductions and observed local latency.
    fidelity=96. if r['psnr_db'] is None else r['psnr_db']
    penalty=max(0,45.-fidelity)/45.
    return round(.67*r['ratio']+.24*penalty+.09*r['ms']/max(baseline_ms,.001),8)


def eval_candidate(volume,gene,state,trials,source_token):
    times=[]; packed=None; decoded=None;delta=None
    for _ in range(trials):
        t=time.perf_counter_ns()
        packed=core.encode3d(volume,gene).raw
        decoded=core.decode3d(packed)
        f,delta=multiscale_field(state,decoded,steps=5)
        times.append((time.perf_counter_ns()-t)/1e6)
        if decoded.shape!=volume.shape or decoded.dtype!=np.uint8 or not np.isfinite(f).all():
            raise AssertionError('reference volume checks failed')
    if core.encode3d(volume,gene).raw!=packed: raise AssertionError('non-deterministic codec output')
    mse,psnr,max_error=core.distortion(volume,decoded)
    return {'gene':list(gene),'valid':True,'ratio':round(len(packed)/volume.nbytes,7),
        'compressed_bytes':len(packed),'raw_bytes':int(volume.nbytes),
        'mse':round(mse,6),'psnr_db':None if math.isinf(psnr) else round(psnr,5),
        'max_error':max_error,'ms':round(statistics.median(times),5),
        'trials_ms':[round(z,5) for z in times], 'codec_sha256':hash_data(packed),
        'field_residual':round(delta[-1],6),'symbolic_token':source_token}


def dominates(a,b):
    """Pareto over size, error, local timing; tolerances prevent noisy equality."""
    av=(a['ratio'],a['mse'],a['ms']);bv=(b['ratio'],b['mse'],b['ms'])
    return all(x<=y+1e-8 for x,y in zip(av,bv)) and any(x<y-1e-8 for x,y in zip(av,bv))


def pareto(rows):
    feasible=[x for x in rows if x.get('score') is not None]
    return [x for x in feasible if not any(dominates(y,x) for y in feasible if y is not x)]


def proposal_set(rng,elite,leaders,limit,visited):
    """A path through the finite Z8 x Z8 x Z12 physical kernel lattice.
    Mathematical septillion exponent is an abstract upper-level address set.
    """
    values=[core.BASELINE,elite]
    for g in leaders[:10]:values.append(g)
    for center in [elite]+leaders[:5]:
        for axis in range(3):
            for direction in (-1,1):
                v=list(center);v[axis]=(v[axis]+direction)%GENE_SHAPE[axis]
                values.append(tuple(v))
    res=[];local=set()
    def add(g):
        g=core.canonical_gene(tuple(g))
        if g not in local and g not in visited:
            local.add(g);res.append(g)
    for v in values:add(v)
    max_available=8*8*12-len(visited)
    stop=min(limit,max_available)
    tries=0
    while len(res)<stop and tries<10000:
        tries+=1
        if rng.random()<.63:
            center=rng.choice([elite]+leaders[:10]) if leaders else elite
            v=list(center)
            for _ in range(rng.choice([1,1,2,3])):
                axis=rng.randrange(3)
                v[axis]=(v[axis]+rng.choice([-2,-1,1,2]))%GENE_SHAPE[axis]
        else:v=[rng.randrange(sz) for sz in GENE_SHAPE]
        add(v)
    return res[:stop]


def audit_candidates(reference,rows):
    for r in rows:
        y=core.decode3d(core.encode3d(reference,tuple(r['gene'])).raw)
        mse,psnr,err=core.distortion(reference,y)
        if not np.array_equal(y,reference) and r['mse']==0:
            raise AssertionError('incorrect lossless designation')
        if abs(mse-r['mse'])>1e-4 or err!=r['max_error']:raise AssertionError('mismatch independent decode')
        if r['psnr_db'] is not None and abs(psnr-r['psnr_db'])>.0001:raise AssertionError('mismatch PSNR')


def run_engine(seed_pdf,input_image,outdir,generations=4,candidates=36,trials=2,psnr_floor=36.5,
               n=24,seed=7,prior_v4_metrics=None):
    outdir=Path(outdir).resolve();outdir.mkdir(parents=True,exist_ok=True)
    seed_pdf=Path(seed_pdf).resolve();source_path=Path(__file__).resolve();base_path=Path(core.__file__).resolve()
    if not seed_pdf.is_file():raise FileNotFoundError(seed_pdf)
    source=source_path.read_bytes();base_source=base_path.read_bytes()
    image_path=Path(input_image).resolve() if input_image else None
    volume=core.sample_volume(image_path,n)
    original_hash=hash_data(volume.tobytes())
    state=volume.astype(np.float32).copy()
    elite=core.BASELINE
    prior_run=None
    if prior_v4_metrics:
        prior_run=json.loads(Path(prior_v4_metrics).read_text())
        if prior_run['input_volume_sha256']!=original_hash:
            raise ValueError('Cannot resume: v4 ground truth differs')
        # Reconstruct v4's recurrent state from its verified elite history.
        for row in prior_run['history']:
            v=core.decode3d(core.encode3d(volume,tuple(row['elite_gene'])).raw)
            state,_=core.inward_field(state,v,steps=4)
        elite=tuple(prior_run['history'][-1]['elite_gene'])
    prior=seed_pdf.read_bytes()
    prev_chain='0'*64
    visited=set();global_candidates=[];history=[]
    start=time.perf_counter()
    for gen in range(1,generations+1):
        tgen=time.perf_counter()
        srccond=hash_data(prior+source+base_source+str(seed).encode())
        drng=random.Random(int(srccond[:16],16)^seed^(gen*7919))
        leaders=[tuple(r['gene']) for r in sorted(pareto(global_candidates),key=lambda a:a['score'])]
        # Always evaluate a fresh exact reference and the incumbent for controlled comparisons.
        available=max(0,candidates-2)
        newgenes=proposal_set(drng,elite,leaders,available,visited|{core.BASELINE,elite})
        genes=list(dict.fromkeys([core.BASELINE,elite]+newgenes))
        if len(genes)<min(candidates,math.prod(GENE_SHAPE)) and not newgenes:
            raise RuntimeError('search exhausted')
        rows=[]
        for i,g in enumerate(genes):
            token=hash_data(canonical({'pdf_source_hash':srccond,'generation':gen,'gene':g,'index':i}))
            rows.append(eval_candidate(volume,g,state,trials,token))
        audit_candidates(volume,rows)
        baseline_ms=rows[0]['ms']
        assert rows[0]['gene']==list(core.BASELINE) and rows[0]['mse']==0
        for r in rows:r['score']=candidate_score(r,baseline_ms,psnr_floor)
        viable=[r for r in rows if r['score'] is not None]
        assert viable
        winner=min(viable,key=lambda x:(x['score'],x['ratio'],x['ms']))
        incumbent=next(r for r in rows if r['gene']==list(elite))
        # Finite-run score improvement must beat a hysteresis threshold.
        accepted=(winner['score']+.002 < incumbent['score']) and tuple(winner['gene'])!=elite
        chosen=winner if accepted else incumbent
        elite=tuple(chosen['gene'])
        for g in newgenes:visited.add(g)
        global_candidates.extend(r for r in rows if r['gene'] not in (list(core.BASELINE),))
        if not quality_gate(chosen,psnr_floor):raise AssertionError('quality gate failed')
        restored=core.decode3d(core.encode3d(volume,elite).raw)
        state,changes=multiscale_field(state,restored,steps=5)
        if state.min()<-.001 or state.max()>255.001:raise AssertionError('field bounds broken')
        pfront=pareto(global_candidates)
        gen_row={
            'generation':gen,'prev_pdf_sha256':hash_data(prior),'source_condition_sha256':srccond,
            'source_sha256':hash_data(source),'base_source_sha256':hash_data(base_source),
            'reference_volume_sha256':original_hash,'candidate_count':len(rows),
            'unique_new_genes':len(newgenes),'cumulative_unique_genes':len(visited),
            'elite_gene':list(elite),'prior_elite_gene':incumbent['gene'],
            'elite_fitness':chosen['score'],'elite_ratio':chosen['ratio'],
            'elite_psnr_db':chosen['psnr_db'],'elite_mse':chosen['mse'],
            'elite_ms':chosen['ms'],'elite_compressed_bytes':chosen['compressed_bytes'],
            'elite_max_error':chosen['max_error'],'promoted':accepted,
            'field_change_linf':round(changes[-1],6),
            'pareto_frontier_count':len(pfront), 'candidates':rows,
            'proven_field_lipschitz_upper_bound':.72,
            'symbolic_address_space':'(10^24)^(10^24) = 10^(24*10^24)'
        }
        report_path=outdir/f'generation_{gen:02d}.pdf'
        generate_pdf(report_path,history+[gen_row],volume,pareto(global_candidates),srccond)
        report=report_path.read_bytes()
        gen_row['pdf_sha256']=hash_data(report)
        chain={'prior_chain':prev_chain,'prior_pdf_sha256':hash_data(prior),
            'pdf_sha256':hash_data(report),'gene':list(elite),'source_condition':srccond}
        gen_row['chain_sha256']=hash_data(canonical(chain))
        data=core.encode3d(volume,elite).raw
        manifest={
            'format':'DM3D5','version':VERSION,'generation':gen,'seed_pdf_sha256':hash_data(prior),
            'report_sha256':hash_data(report),'v5_source_sha256':hash_data(source),
            'v4_source_sha256':hash_data(base_source),'volume_sha256':original_hash,
            'encoded_sha256':hash_data(data),'source_condition_sha256':srccond,
            'chain_sha256':gen_row['chain_sha256'], 'search_space':'10^(24*10^24)',
            'candidates_tested':len(rows)}
        archive=outdir/f'generation_{gen:02d}.dm3d5'
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
            for name,data2 in [('manifest.json',canonical(manifest)),('previous_generation.pdf',prior),
                ('current_report.pdf',report),('v5_runtime_source.py',source),('v4_codec_source.py',base_source),
                ('reference_volume.bin',volume.tobytes()),('encoded_volume.dm3d',data),
                ('generation_metrics.json',canonical(gen_row))]:z.writestr(name,data2)
        gen_row['archive_sha256']=hash_data(archive.read_bytes())
        gen_row['wall_ms']=round((time.perf_counter()-tgen)*1e3,3)
        (outdir/f'generation_{gen:02d}_audit.json').write_text(json.dumps(gen_row,indent=2)+'\n')
        history.append(gen_row);prev_chain=gen_row['chain_sha256'];prior=report
        print(f"GEN={gen} EVALUATED={len(rows)} UNIQUE={len(visited)} ELITE={elite} PSNR={gen_row['elite_psnr_db']} RATIO={gen_row['elite_ratio']:.5f} FRONT={len(pfront)} VERIFIED",flush=True)
    result={'version':VERSION,'cardinality':'(10^24)^(10^24)','log10_cardinality':str(24*10**24),
        'genotype_lattice':list(GENE_SHAPE),'physical_genotype_count':math.prod(GENE_SHAPE),
        'evaluations':sum(x['candidate_count'] for x in history),'unique_nonreference_candidates':len(visited),
        'generations':len(history),'voxel_shape':list(volume.shape),'volume_sha256':original_hash,
        'seed_pdf_sha256':hash_data(seed_pdf.read_bytes()),'user_seed':seed,'psnr_floor':psnr_floor,'v4_resume':bool(prior_run),
        'wall_seconds':round(time.perf_counter()-start,3),
        'last_pdf':str(outdir/f'generation_{generations:02}.pdf'),
        'history':history,
        'interpretation':'Finite local sampled kernel autotuning, not septillion iteration execution, neural learning, source rewriting or global convergence proof.'}
    (outdir/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    verify(outdir,seed_pdf)
    return result


def tt(c,x,y,s,size=10,color=WHITE,bold=False):
    c.setFont('Helvetica-Bold' if bold else 'Helvetica',size)
    c.setFillColor(color);c.drawString(x,y,str(s))


def rect(c,x,y,w,h,border=BLUE):
    c.setFillColor(PANEL);c.setStrokeColor(border);c.setLineWidth(.8)
    c.roundRect(x,y,w,h,9,fill=1,stroke=1)


def line(c,x,y,x2,y2,color=BLUE):
    c.setStrokeColor(color);c.setLineWidth(1.5);c.line(x,y,x2,y2)


def frame(c,n,title,sub):
    c.setFillColor(BG);c.rect(0,0,842,595,fill=1,stroke=0)
    tt(c,37,558,'DM3D  /  V5  /  SELF-CONDITIONED VERIFIED 3D ENGINE',10,BLUE,True)
    tt(c,37,518,title,22,WHITE,True)
    tt(c,37,494,sub,9,LIGHT)
    line(c,37,480,801,480)
    line(c,37,35,801,35)
    tt(c,37,21,'MATHEMATICAL CAPACITY IS SYMBOLIC  /  ALL DISPLAYED COUNTS ARE EXECUTED',8,LIGHT)
    tt(c,751,21,f'{n:02d} / 06',9,BLUE,True)


def plot_vol(c,v,base_x=290,base_y=120,scale=7.1):
    core.iso_points(c,v,x0=base_x,y0=base_y,scale=scale,limit=390)


def generate_pdf(path,history,volume,front,condition_hash):
    path=Path(path)
    c=canvas.Canvas(str(path),pagesize=(842,595),pageCompression=1,invariant=1)
    row=history[-1];total=sum(x['candidate_count'] for x in history)
    frame(c,1,'Inward-looped in three spatial dimensions','Multiscale field contraction, tested kernel evolution, and PDF/source-conditioned recurrence')
    labels=[('3D INPUT','Immutable XYZ bytes'),('CODEC','Encode / decode'),('INWARD','5 multiscale steps'),
        ('CTR','Validate + benchmark'),('SELECT','Pareto + score'),('FEEDBACK','PDF / source / hash')]
    for i,(heading,detail) in enumerate(labels):
        x=39+i*132;rect(c,x,288,121,127)
        tt(c,x+9,380,f'{i+1:02}  {heading}',9.3,BLUE,True)
        tt(c,x+9,350,detail,8.5,WHITE)
        if i<5:tt(c,x+119,336,'>',12,GREEN,True)
    rect(c,40,92,760,164)
    vals=[('EXECUTED EVALS',str(total)),('3D GRID',f'{volume.shape[0]} x {volume.shape[0]} x {volume.shape[0]}'),
      ('SELECTED GENE',str(tuple(row['elite_gene']))),('PSNR',f"{row['elite_psnr_db']:.2f} dB" if row['elite_psnr_db'] is not None else 'exact'),
      ('CODEC / RAW',f"{row['elite_ratio']:.3f}x"),('PARETO FRONT',str(row['pareto_frontier_count']))]
    for i,(k,v) in enumerate(vals):
        x=62+(i%3)*247;y=211-(i//3)*70
        tt(c,x,y,k,9,BLUE,True);tt(c,x,y-25,v,16,WHITE,True)
    c.showPage()

    frame(c,2,'3D operator: genuine XYZ geometry','2 x 2 x 2 coarse lattice, sparse residual correction and multiresolution contraction')
    rect(c,40,86,489,367)
    plot_vol(c,volume,base_x=280,base_y=132,scale=7.4)
    rect(c,543,86,255,367)
    phrases=['X is uint8[n,n,n]','3-axis toroidal roll','Latent: XYZ / 2 each','2 x 2 x 2 residuals','Verified decoder','Nearest-neighbor 3D avg','Coarse block projection','5 state update steps']
    for i,p in enumerate(phrases):tt(c,555,423-i*29,p,10,GREEN if i in (0,5) else WHITE)
    tt(c,554,167,"F' = .55 F + .10 N3(F)",9.1,BLUE,True)
    tt(c,554,148,'    + .07 U(D(F)) + .28 X',9.1,WHITE)
    tt(c,554,119,'Lipschitz <= 0.72',9.7,GOLD,True)
    c.showPage()

    frame(c,3,'Evolution history and bounded computation','Each candidate undergoes native NumPy codec, decoding, state recurrence and quality checks')
    rect(c,40,153,759,299)
    cols=[55,103,223,316,408,500,607,696]
    hdr=['GEN','GENOME','EVAL','PSNR','SIZE/RAW','LOCAL ms','PARETO','PROMOTE']
    for x,h in zip(cols,hdr):tt(c,x,418,h,9,BLUE,True)
    line(c,52,405,787,405)
    for j,r in enumerate(history[-6:]):
        y=373-37*j
        values=[r['generation'],str(tuple(r['elite_gene'])),r['candidate_count'],
            'exact' if r['elite_psnr_db'] is None else f"{r['elite_psnr_db']:.2f}",
            f"{r['elite_ratio']:.3f}",f"{r['elite_ms']:.2f}",r['pareto_frontier_count'],
            'YES' if r['promoted'] else 'NO']
        for x,value in zip(cols,values):tt(c,x,y,value,9, GREEN if x==cols[-1] and r['promoted'] else WHITE)
        line(c,53,y-11,787,y-11,GRID)
    tt(c,43,125,'Local times are median encode + decode + 5-step 3D field; not wall-to-wall report time.',9,LIGHT)
    tt(c,43,104,'Optimizer ranks size, PSNR and time; quality threshold is mandatory.',9,LIGHT)
    tt(c,43,83,f"Evaluations: {total}; symbolic septillion^septillion configurations never enumerated.",9,GOLD)
    c.showPage()

    frame(c,4,'The inward recursive feedback boundary','Previously generated PDF bytes and both runtime sources determine the next proposal stream')
    flow=[('PREVIOUS PDF','input bytes'),('SOURCE STATE','v5 + v4 bytes'),('HASH CONTEXT','SHA-256 seed'),
          ('NEW CANDIDATES','3D toroidal moves'),('VERIFY / SELECT','error + size + ms'),('NEXT PDF','new state')]
    for i,(head,detail) in enumerate(flow):
        x=54+(i%3)*260;y=320-(i//3)*160
        rect(c,x,y,231,118)
        tt(c,x+17,y+79,head,11,BLUE,True);tt(c,x+17,y+48,detail,10,LIGHT)
        if i not in (2,5):tt(c,x+213,y+17,'>',15,GREEN,True)
    tt(c,54,71,'Proposal conditioning digest: '+condition_hash[:51]+'...',9,GOLD)
    c.showPage()

    frame(c,5,'Symbolic vastness versus physical search','The address-space expression is exact; finite measured trials are the only computation claimed')
    rect(c,46,273,752,174)
    tt(c,66,398,'S = 10^24',17,WHITE,True)
    tt(c,66,357,'P = S^S = 10^(24 x 10^24)',20,BLUE,True)
    tt(c,66,315,'log10(P) = 24 x 10^24',15,GREEN,True)
    rect(c,46,93,752,146)
    tt(c,64,208,'Small kernel genotype space: Z8 x Z8 x Z12 = 768 configurations.',11,WHITE)
    tt(c,64,173,f"Unique new physical genomes sampled: {row['cumulative_unique_genes']}",11,GREEN)
    tt(c,64,138,'SHA-256 tokens identify tested proposals; they are not uniform samples of P.',10,LIGHT)
    tt(c,64,111,'No P-sized array, address index, loop, or exhaustive global optimum exists.',10,GOLD)
    c.showPage()

    frame(c,6,'Audit and reproducible run evidence','Previous PDF, both source files, immutable input, selected codec, and metrics are archived')
    rect(c,43,205,755,240)
    for i,r in enumerate(history[-5:]):
        y=412-i*42
        tt(c,59,y,f"GEN {r['generation']:02d}",10,GREEN,True)
        tt(c,153,y,'parent: '+r['prev_pdf_sha256'][:21]+'...',9,LIGHT)
        if r.get('pdf_sha256'):tt(c,485,y,'output: '+r['pdf_sha256'][:21]+'...',9,WHITE)
    rect(c,43,78,755,101)
    for i,s in enumerate(['Lossless reference (7,0,0) always independently tested.',
         'Selected codec is lossy unless MSE=0; PSNR is measured against immutable source.',
         'This is configuration self-optimisation, not training or autonomous source rewriting.']):
        tt(c,58,150-i*27,s,10,GREEN if i==0 else LIGHT)
    c.save()
    with fitz.open(path) as d:
        if len(d)!=6 or any(len(p.get_text())<80 for p in d):
            raise RuntimeError('generated report failed structure checks')


def verify(outdir,seedpdf):
    outdir=Path(outdir);seedpdf=Path(seedpdf)
    result=json.loads((outdir/'metrics.json').read_text())
    prev=seedpdf.read_bytes();prev_chain='0'*64
    source=Path(__file__).read_bytes();base=Path(core.__file__).read_bytes()
    for row in result['history']:
        k=row['generation'];a=outdir/f'generation_{k:02d}.dm3d5';p=outdir/f'generation_{k:02d}.pdf'
        curr=p.read_bytes()
        with zipfile.ZipFile(a) as f:
            m=json.loads(f.read('manifest.json'))
            checks={'previous_generation.pdf':prev,'current_report.pdf':curr,'v5_runtime_source.py':source,'v4_codec_source.py':base}
            for file,expected in checks.items():
                if f.read(file)!=expected:raise AssertionError(file+' bytes incorrect')
            ref=f.read('reference_volume.bin');enc=f.read('encoded_volume.dm3d')
            if m['volume_sha256']!=hash_data(ref) or m['encoded_sha256']!=hash_data(enc):raise AssertionError('data hash invalid')
            if m['report_sha256']!=hash_data(curr) or m['seed_pdf_sha256']!=hash_data(prev):raise AssertionError('PDF integrity invalid')
            if m['source_condition_sha256']!=hash_data(prev+source+base+str(result['user_seed']).encode()):
                raise AssertionError('bad source conditioning')
            x=np.frombuffer(ref,dtype=np.uint8).reshape(result['voxel_shape'])
            y=core.decode3d(enc)
            mse,psnr,err=core.distortion(x,y)
            if abs(mse-row['elite_mse'])>1e-4 or err!=row['elite_max_error']:
                raise AssertionError('reconstruction mismatch')
            if y.shape!=x.shape or (row['elite_psnr_db'] is not None and row['elite_psnr_db']<result['psnr_floor']):
                raise AssertionError('decoded shape or PSNR gate invalid')
            if tuple(json.loads(f.read('generation_metrics.json'))['elite_gene'])!=tuple(row['elite_gene']):raise AssertionError('metrics mismatch')
            chain={'prior_chain':prev_chain,'prior_pdf_sha256':hash_data(prev),'pdf_sha256':hash_data(curr),
                'gene':row['elite_gene'],'source_condition':row['source_condition_sha256']}
            if hash_data(canonical(chain))!=row['chain_sha256'] or m['chain_sha256']!=row['chain_sha256']:
                raise AssertionError('chain invalid')
        with fitz.open(p) as pdf:
            if len(pdf)!=6:raise AssertionError('bad PDF pages')
        prev=curr;prev_chain=row['chain_sha256']
    return len(result['history'])


def selftest():
    rng=np.random.default_rng(77)
    v=rng.integers(0,256,(16,16,16),dtype=np.uint8)
    assert np.array_equal(core.decode3d(core.encode3d(v,core.BASELINE).raw),v)
    for gene in [(3,4,9),(0,0,0),(7,0,11),(2,3,2)]:
        y=core.decode3d(core.encode3d(v,gene).raw)
        assert y.shape==v.shape
    a=rng.uniform(0,255,size=v.shape).astype(np.float32)
    b=rng.uniform(0,255,size=v.shape).astype(np.float32)
    z=np.zeros_like(a)
    f,_=multiscale_field(a,z,steps=5);g,_=multiscale_field(b,z,steps=5)
    assert np.max(np.abs(f-g)) <= (.72005**5)*np.max(np.abs(a-b))+1e-3
    rows=[dict(ratio=1.0,mse=0.,ms=2.,score=1.,gene=[7,0,0]),dict(ratio=.5,mse=3.,ms=1.,score=.6,gene=[2,3,2])]
    assert len(pareto(rows))==2
    print('PASS: lossless reference, genuine XYZ codec, multiscale contraction, Pareto gate')


def main():
    p=argparse.ArgumentParser(description='DM3D v5: verified finite inward recurrence over symbolic septillion^septillion')
    s=p.add_subparsers(dest='action',required=True)
    s.add_parser('selftest')
    e=s.add_parser('evolve');e.add_argument('--seed-pdf',required=True);e.add_argument('--input-image',default='')
    e.add_argument('--prior-v4-metrics',default='');e.add_argument('--outdir',default='dm3d_v5_run')
    e.add_argument('--generations',type=int,default=4);e.add_argument('--candidates',type=int,default=36)
    e.add_argument('--trials',type=int,default=2);e.add_argument('--size',type=int,default=24)
    e.add_argument('--psnr',type=float,default=36.5);e.add_argument('--seed',type=int,default=7)
    v=s.add_parser('verify');v.add_argument('--seed-pdf',required=True);v.add_argument('--outdir',required=True)
    a=p.parse_args()
    if a.action=='selftest':selftest();return
    if a.action=='verify':print('PASS',verify(a.outdir,a.seed_pdf),'generations verified');return
    if not(1<=a.generations<=10 and 3<=a.candidates<=128 and 1<=a.trials<=10 and 8<=a.size<=40 and a.size%2==0 and 0<=a.psnr<=90):
        p.error('invalid bounded execution parameters')
    r=run_engine(a.seed_pdf,a.input_image,a.outdir,a.generations,a.candidates,a.trials,a.psnr,a.size,a.seed,a.prior_v4_metrics or None)
    print(json.dumps({'passed':True,'evaluations':r['evaluations'],'last_gene':r['history'][-1]['elite_gene'],
        'report':r['last_pdf'],'elapsed_seconds':r['wall_seconds']},indent=2))

if __name__=='__main__':main()
