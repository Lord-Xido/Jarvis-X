#!/usr/bin/env python3
"""Bounded outer-loop DM3D architecture search over topology and latent width.

Only candidate architectures that meet the fixed development-validation target
are eligible for the minimum measured packet-size selection. Independent test
is evaluated after selection; it is not used to pick the winner.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict,replace
import json
import math
from pathlib import Path
from dr_moagi_vnext import (Config,Swarm,pyramid,candidate_search,demo_data,run)


def evolve_architecture(cfg:Config,x,target_mse:float,max_candidates=6):
    cfg.validate()
    if max_candidates<1 or max_candidates>20:
        raise ValueError("candidate budget must be between 1 and 20")
    choices=[]
    depth_options=sorted({cfg.levels, max(1,cfg.levels-1), cfg.levels+1})
    width_options=sorted({cfg.hidden,max(4,cfg.hidden//2)})
    for depth in depth_options:
        if cfg.edge%(2**depth) or cfg.edge//(2**depth)<2:continue
        for width in width_options:
            choices.append((depth,width))
    choices=choices[:max_candidates]
    if cfg.tiles>=4:
        nval=max(1,(cfg.tiles-1)//4)
        training=x[:-1-nval];validation=x[-1-nval:-1]
    else:
        training=x[:1];validation=x[1:]
    evaluations=[]
    for depth,width in choices:
        variant=replace(cfg,levels=depth,hidden=width,
                        train_steps=min(cfg.train_steps,40))
        m=Swarm(variant)
        tr,_=pyramid(training,depth)
        val,_=pyramid(validation,depth)
        training_score=m.train(tr,val)
        selected,_=candidate_search(m,validation,target_mse)
        evaluations.append({'levels':depth,'hidden':width,
                            'validation_mse':selected['mse'],
                            'validation_packet_bytes':selected['bytes'],
                            'meets_target':selected['meets_target'],
                            'detail_keep':selected['keep'],
                            'base_correction':selected['base_correction'],
                            'train_accepted':training_score['accepted'],
                            'train_rejected':training_score['rejected']})
    feasible=[v for v in evaluations if v['meets_target']]
    winner=min(feasible,key=lambda r:(r['validation_packet_bytes'],r['validation_mse'])) \
           if feasible else min(evaluations,key=lambda r:(r['validation_mse'],r['validation_packet_bytes']))
    best=replace(cfg,levels=winner['levels'],hidden=winner['hidden'])
    return best,winner,evaluations


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--edge',type=int,default=8)
    p.add_argument('--tiles',type=int,default=8)
    p.add_argument('--levels',type=int,default=2)
    p.add_argument('--hidden',type=int,default=12)
    p.add_argument('--steps',type=int,default=80)
    p.add_argument('--candidates',type=int,default=6)
    p.add_argument('--target-mse',type=float,default=.015)
    p.add_argument('--out',type=Path,default=Path('dm3d_evolved_output'))
    a=p.parse_args(argv)
    cfg=Config(edge=a.edge,tiles=a.tiles,levels=a.levels,hidden=a.hidden,train_steps=a.steps)
    x=demo_data(cfg)
    chosen,winner,table=evolve_architecture(cfg,x,a.target_mse,a.candidates)
    result=run(chosen,a.out,a.target_mse,x)
    payload={'selected_architecture':{'levels':chosen.levels,'hidden':chosen.hidden},
             'selection_evidence':winner,'all_candidates':table,
             'final_run':result}
    (a.out/'evolution_search.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(json.dumps(payload,indent=2))

if __name__=='__main__':main()
