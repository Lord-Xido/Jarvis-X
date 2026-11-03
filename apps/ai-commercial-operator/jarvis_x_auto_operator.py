#!/usr/bin/env python3
"""Jarvis X Auto Operator 1.0 — auditable local-to-global AI 4P planning.

Runs an actual small MLP backtest and an exact discrete 4P portfolio optimiser.
NO external deployment, billing, order execution, advertising, or messaging.
All generated actions are proposals requiring human authorization. Stdlib only.

Usage:
  python3 jarvis_x_auto_operator.py --demo --output outputs
  python3 jarvis_x_auto_operator.py --data observations.csv --output outputs
  python3 jarvis_x_auto_operator.py --demo --watch-seconds 3600

CSV schema (one row per market/week, ascending weeks):
  week,market_id,price_zar,promotion_zar,place_reach,product_quality_index,
  affordability_index,sentiment,previous_new_paid,new_paid,uncensored
To interpret new_paid as unconstrained demand, uncensored must be true. Rows
marked censored are excluded from model fitting and holdout tests.
Market base rates, initial active accounts, retention and capacity are in config.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import html
import itertools
import json
import math
import os
from pathlib import Path
import random
import statistics
import time

HERE = Path(__file__).resolve().parent
FIELDS = ['week','market_id','price_zar','promotion_zar','place_reach',
          'product_quality_index','affordability_index','sentiment',
          'previous_new_paid','new_paid','uncensored']
FEATURE_NAMES = ['log_market_size','log_price_ratio','log_promotion',
                 'place_reach','product_quality','affordability','sentiment',
                 'previous_new_ratio','season_sin','season_cos']


def clamp(x, lo, hi): return max(lo, min(hi, x))


def fmt_zar(x): return ('−' if x < 0 else '') + 'R' + format(abs(x), ',.0f')


def hash_data(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def make_demo(config, output):
    """Deterministic historical SYNTHETIC observations with no stockout censoring."""
    rng=random.Random(20261103)
    rows=[]
    for m in config['markets']:
        last=m['market_baseline_new_week']
        memory=0.
        for week in range(1,53):
            price=m['base_weekly_price_zar']*(0.85+0.3*(week%13)/12)
            promo=m['marketing_menu_zar'][week%3]
            memory=0.45*memory+promo
            reach=m['place_reach']*(0.96+0.08*math.sin(week/11))
            quality=m['product_quality_index']
            afford=m['affordability_index']
            sentiment=clamp(m['sentiment']+0.08*math.sin(week*.18),-.5,.5)
            seasonal=1+0.12*math.sin(2*math.pi*week/52)
            base=m['market_baseline_new_week']
            # Promo impact intentionally decreases with spending (saturation).
            exposure=1+0.65*memory/(8000+memory)
            latent=base*(reach/.5)*(quality**1.3)*(afford/.7) * \
                   (price/m['base_weekly_price_zar'])**(-1.3) * \
                   exposure*(1+0.3*sentiment)*seasonal
            new_paid=max(0,round(latent+rng.gauss(0,max(1,latent*.12))))
            rows.append(dict(week=week,market_id=m['id'],price_zar=round(price,2),
                promotion_zar=promo,place_reach=round(reach,5),
                product_quality_index=quality,affordability_index=afford,
                sentiment=round(sentiment,5),previous_new_paid=last,
                new_paid=new_paid,uncensored=True))
            last=new_paid
    output.mkdir(parents=True,exist_ok=True)
    with (output/'synthetic_observations.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def read_csv(path,config):
    known={m['id'] for m in config['markets']}
    rows=[]
    with Path(path).open(newline='') as f:
        r=csv.DictReader(f)
        missing=set(FIELDS)-set(r.fieldnames or [])
        if missing: raise ValueError('Missing required CSV columns: '+','.join(sorted(missing)))
        for line,x in enumerate(r,2):
            if x['market_id'] not in known: raise ValueError(f'Unknown market on line {line}: {x["market_id"]}')
            data={k:(int(x[k]) if k in ('week','previous_new_paid','new_paid')
                     else x[k].strip().lower() in ('1','true','yes') if k=='uncensored'
                     else x[k] if k=='market_id' else float(x[k])) for k in FIELDS}
            if data['week'] < 1 or data['new_paid'] < 0 or data['previous_new_paid'] < 0:
                raise ValueError(f'Invalid negative count or week on line {line}')
            if data['price_zar'] <= 0 or data['promotion_zar'] < 0 or not 0<=data['place_reach']<=1:
                raise ValueError(f'Invalid price/promotion/reach on line {line}')
            if data['product_quality_index']<=0 or not 0<=data['affordability_index']<=1:
                raise ValueError(f'Invalid utility/affordability on line {line}')
            if not all(math.isfinite(v) for v in data.values() if isinstance(v,(int,float))):
                raise ValueError(f'Nonfinite numeric value on line {line}')
            rows.append(data)
    if len(rows)<80: raise ValueError('At least 80 rows are needed for demo MLP cross-market fitting.')
    keys=[(x['market_id'],x['week']) for x in rows]
    if len(set(keys))!=len(keys): raise ValueError('Duplicate market/week observations')
    rows.sort(key=lambda x:(x['week'],x['market_id']))
    return rows


def features(r,m):
    base=max(1,m['market_baseline_new_week'])
    return [math.log1p(base),math.log(r['price_zar']/m['base_weekly_price_zar']),
            math.log1p(r['promotion_zar']/1000),r['place_reach'],
            r['product_quality_index'],r['affordability_index'],r['sentiment'],
            r['previous_new_paid']/base,math.sin(2*math.pi*r['week']/52),
            math.cos(2*math.pi*r['week']/52)]


class TinyANN:
    """10→14 tanh→1, full-batch gradient with Adam, train-only normalisation."""
    def __init__(self,seed=7,width=14):
        rng=random.Random(seed)
        self.width=width
        self.w=[[rng.gauss(0,0.22) for _ in FEATURE_NAMES] for _ in range(width)]
        self.b=[0.]*width
        self.v=[rng.gauss(0,.15) for _ in range(width)]
        self.c=0.
        self.mean=None;self.std=None;self.ymean=None;self.ystd=None

    def norm(self,x):
        return [(xi-u)/s for xi,u,s in zip(x,self.mean,self.std)]

    def _forward(self,x):
        a=[math.tanh(sum(wj*xj for wj,xj in zip(w,x))+b) for w,b in zip(self.w,self.b)]
        return sum(v*a1 for v,a1 in zip(self.v,a))+self.c,a

    def predict(self,x):
        z,_=self._forward(self.norm(x))
        return max(0.,math.expm1(clamp(z*self.ystd+self.ymean,-8,13)))

    def _params(self):
        return [v for row in self.w for v in row]+self.b+self.v+[self.c]

    def _setparams(self,p):
        n=len(FEATURE_NAMES);w=self.width
        self.w=[p[i*n:(i+1)*n] for i in range(w)]
        j=w*n;self.b=p[j:j+w];j+=w;self.v=p[j:j+w];self.c=p[-1]

    def train(self,train,val,market_map,epochs=350):
        raw=[features(r,market_map[r['market_id']]) for r in train]
        self.mean=[statistics.mean(x[k] for x in raw) for k in range(len(FEATURE_NAMES))]
        self.std=[max(.03,statistics.pstdev(x[k] for x in raw)) for k in range(len(FEATURE_NAMES))]
        y=[math.log1p(r['new_paid']) for r in train]
        self.ymean=statistics.mean(y);self.ystd=max(.2,statistics.pstdev(y))
        xs=[self.norm(x) for x in raw]
        ys=[(v-self.ymean)/self.ystd for v in y]
        p=self._params();m1=[0.]*len(p);m2=[0.]*len(p)
        best=p[:];best_score=float('inf');best_epoch=0
        val_x=[features(r,market_map[r['market_id']]) for r in val]
        for epoch in range(1,epochs+1):
            grads=[0.]*len(p)
            for x,y0 in zip(xs,ys):
                yhat,a=self._forward(x);err=clamp(yhat-y0,-8,8)
                for j in range(self.width):
                    s=err*self.v[j]*(1-a[j]*a[j]);offset=j*len(x)
                    for k in range(len(x)): grads[offset+k]+=s*x[k]
                bstart=self.width*len(x)
                for j in range(self.width):
                    grads[bstart+j]+=err*self.v[j]*(1-a[j]*a[j])
                    grads[bstart+self.width+j]+=err*a[j]
                grads[-1]+=err
            rate=.014/(1+.001*epoch)
            for k in range(len(p)):
                grad=grads[k]/len(xs)+.00005*p[k]
                m1[k]=.9*m1[k]+.1*grad
                m2[k]=.999*m2[k]+.001*grad*grad
                p[k]-=rate*(m1[k]/(1-.9**epoch))/(math.sqrt(m2[k]/(1-.999**epoch))+1e-8)
            self._setparams(p)
            if epoch%5==0 or epoch==epochs:
                score=statistics.mean(abs(self.predict(x)-r['new_paid']) for x,r in zip(val_x,val))
                if score<best_score:best_score=score;best=p[:];best_epoch=epoch
        self._setparams(best)
        return best_epoch,best_score


def evaluate(rows,market_map):
    weeks=sorted({r['week'] for r in rows})
    if len(weeks)<15: raise ValueError('Need at least 15 chronological weeks')
    cut_train=weeks[math.floor(.65*len(weeks))-1]
    cut_val=weeks[math.floor(.8*len(weeks))-1]
    train=[r for r in rows if r['week']<=cut_train and r['uncensored']]
    val=[r for r in rows if cut_train<r['week']<=cut_val and r['uncensored']]
    test=[r for r in rows if r['week']>cut_val and r['uncensored']]
    if min(len(train),len(val),len(test))<10:
        raise ValueError('Insufficient uncensored training, validation or test observations')
    model=TinyANN()
    best_epoch,val_mae=model.train(train,val,market_map)
    absolute=[];baseline=[];test_predictions=[]
    for r in test:
        estimate=model.predict(features(r,market_map[r['market_id']]))
        absolute.append(abs(estimate-r['new_paid']))
        baseline.append(abs(r['previous_new_paid']-r['new_paid']))
        test_predictions.append({'market_id':r['market_id'],'week':r['week'],
                                 'observed_new_paid':r['new_paid'],'predicted':round(estimate,2)})
    mae=statistics.mean(absolute);naive=statistics.mean(baseline)
    calibration_abs=[abs(model.predict(features(r,market_map[r['market_id']]))-r['new_paid']) for r in val]
    calibration_abs.sort();idx=min(len(calibration_abs)-1,math.ceil(.9*(len(calibration_abs)+1))-1)
    halfwidth=calibration_abs[idx]
    actual_cov=sum(v<=halfwidth for v in absolute)/len(absolute)
    mean_new=statistics.mean(r['new_paid'] for r in test)
    rel=mae/max(1.,mean_new)
    return model,{'train_rows':len(train),'validation_rows':len(val),'test_rows':len(test),
                 'train_last_week':cut_train,'val_last_week':cut_val,
                 'best_epoch':best_epoch,'validation_mae':round(val_mae,3),
                 'test_mae':round(mae,3),'lag_naive_test_mae':round(naive,3),
                 'relative_test_mae':round(rel,4),
                 'nominal_interval_coverage':.9,'actual_test_coverage':round(actual_cov,4),
                 'interval_half_width_customers':round(halfwidth,3),
                 'holdout_predictions':test_predictions}


def plan(config,rows,model,metrics,source_mode):
    market_map={m['id']:m for m in config['markets']}
    last_week=max(r['week'] for r in rows);forecast_week=last_week+1
    latest={}
    for r in rows:
        if r['market_id'] not in latest or r['week']>latest[r['market_id']]['week']:
            latest[r['market_id']]=r
    absent=set(market_map)-set(latest)
    if absent:raise ValueError('Missing recent observations for markets: '+str(absent))
    model_ok=metrics['relative_test_mae']<=config['model_max_relative_mae'] and \
             metrics['test_mae']<=config['model_max_naive_mae_ratio']*max(.0001,metrics['lag_naive_test_mae'])
    decisions=[]
    candidate_lists=[]
    for m in config['markets']:
        last=latest[m['id']]
        survivor=m['active_accounts']*(1-m['weekly_churn'])
        choices=[]
        for price_factor,promotion in itertools.product((.85,1.,1.15),m['marketing_menu_zar']):
            price=round(m['base_weekly_price_zar']*price_factor,2)
            new_row=dict(week=forecast_week,market_id=m['id'],price_zar=price,
                         promotion_zar=promotion,place_reach=m['place_reach'],
                         product_quality_index=m['product_quality_index'],
                         affordability_index=m['affordability_index'],sentiment=m['sentiment'],
                         previous_new_paid=last['new_paid'])
            prediction=model.predict(features(new_row,m))
            if not model_ok:
                # Hold spending; conservative fallback not used for automatic actions.
                prediction=max(0.,last['new_paid'])
            expected_new=min(m['qualified_prospects_remaining'],max(0,round(prediction)))
            accounts=survivor+expected_new
            requests=math.ceil(accounts*config['requests_per_active_account_week'])
            revenue=survivor*m['base_weekly_price_zar']+expected_new*price
            compute_cost=requests*config['variable_compute_cost_zar_per_request']
            support_cost=accounts*config['support_cash_zar_per_active_account_week']
            variable_cost=compute_cost+support_cost
            contrib=revenue-variable_cost-promotion
            choices.append({'market_id':m['id'],'market':m['label'],'price_zar':price,
                            'promo_zar':promotion,'forecast_new_paid':expected_new,
                            'existing_survivors':round(survivor,2),
                            'forecast_active_paid':round(accounts,2),
                            'requests':requests,'weekly_revenue_zar':round(revenue,2),
                            'variable_cash_cost_zar':round(variable_cost,2),
                            'contribution_after_marketing_zar':round(contrib,2),
                            'product_quality':m['product_quality_index'],
                            'place_reach':m['place_reach']})
        if not model_ok:
            choices=[c for c in choices if c['promo_zar']==0 and
                     math.isclose(c['price_zar'],m['base_weekly_price_zar'])]
        candidate_lists.append(choices)
    # Exact multiple-choice combinatorial search over allowed 4P menus.
    selected=None;best_profit=-math.inf
    for allocation in itertools.product(*candidate_lists):
        spent=sum(x['promo_zar'] for x in allocation)
        requests=sum(x['requests'] for x in allocation)
        contribution=sum(x['contribution_after_marketing_zar'] for x in allocation)
        revenue=sum(x['weekly_revenue_zar'] for x in allocation)
        variable=sum(x['variable_cash_cost_zar'] for x in allocation)
        fixed=config['weekly_platform_fixed_cash_zar']
        receivables=config['opening_receivables_zar']
        collections=config['cash_collection_fraction_week']*revenue+receivables
        cash_after=config['starting_cash_zar']+collections-variable-spent-fixed
        if spent>config['weekly_marketing_budget_zar'] or requests>config['compute_request_capacity_week']:
            continue
        if cash_after<config['minimum_cash_reserve_zar']:
            continue
        profit=contribution-fixed
        if profit>best_profit:
            selected=allocation;best_profit=profit
    if selected is None:
        return {'status':'INFEASIBLE','execution':'PROPOSALS_ONLY','model_qualified':model_ok,
                'reason':'No feasible plan satisfies marketing, compute and cash-reserve limits.',
                'model_metrics':metrics}
    selected=list(selected)
    revenue=sum(x['weekly_revenue_zar'] for x in selected)
    marketing=sum(x['promo_zar'] for x in selected)
    variable=sum(x['variable_cash_cost_zar'] for x in selected)
    expected_collections=config['cash_collection_fraction_week']*revenue+config['opening_receivables_zar']
    closing_cash=config['starting_cash_zar']+expected_collections-variable-marketing-config['weekly_platform_fixed_cash_zar']
    result={'status':'PLANNED','execution':'PROPOSALS_ONLY','data_mode':source_mode,
            'model_qualified':model_ok,'asof_week':last_week,'forecast_week':forecast_week,
            'decision_policy':'EXACT_DISCRETE_MENU_OPTIMISATION; NOT A CAUSAL CAMPAIGN ESTIMATE',
            'markets':selected,'model_metrics':{k:v for k,v in metrics.items() if k!='holdout_predictions'},
            'total_new_paid_forecast':sum(x['forecast_new_paid'] for x in selected),
            'weekly_revenue_zar':round(revenue,2),'weekly_inference_requests':sum(x['requests'] for x in selected),
            'weekly_promotion_zar':marketing,'weekly_variable_cash_zar':round(variable,2),
            'weekly_platform_fixed_cash_zar':config['weekly_platform_fixed_cash_zar'],
            'weekly_operating_contribution_zar':round(best_profit,2),
            'forecast_customer_collections_zar':round(expected_collections,2),
            'forecast_end_cash_zar':round(closing_cash,2),
            'forecast_end_receivables_zar':round((1-config['cash_collection_fraction_week'])*revenue,2),
            'compute_utilisation':round(sum(x['requests'] for x in selected)/config['compute_request_capacity_week'],4),
            'constraints':{'marketing_limit_zar':config['weekly_marketing_budget_zar'],
                           'compute_limit_requests':config['compute_request_capacity_week'],
                           'cash_reserve_zar':config['minimum_cash_reserve_zar']},
            'warnings':[
                'All market sizes, unit economics and demo observations are synthetic unless replaced with independently verified data.',
                'ANN forecasts are observational and may not identify causal effects of promotion or pricing.',
                'Market expansion and pricing changes require legal, financial and customer approval.',
                'Compute capacity is request-equivalent only; production requires latency, token, GPU and burst-load measurements.',
                'No live ads, cloud resources, customer records, payments, or share orders were accessed or modified.']}
    if not model_ok:
        result['status']='HOLD_FORECAST_VALIDITY'
        result['warnings'].append('MLP failed backtest acceptance thresholds. Marketing is forced to zero and base prices retained.')
    elif best_profit <= 0:
        result['status']='HOLD_UNIT_ECONOMICS'
        result['warnings'].append('Negative expected weekly operating contribution: do not scale or commit marketing/infrastructure capital on this plan.')
    result['expansion_authorized']=False
    result['approval_gate']={'human_approval_required':True,'external_execution_enabled':False,
                             'expansion_ready':False,
                             'reason':'No external connector or contract authorisation; negative economics/uncertainty may impose additional holds.'}
    return result


def write_report(output,result,config,source_digest):
    output.mkdir(parents=True,exist_ok=True)
    result['input_sha256']=source_digest
    output.joinpath('latest_plan.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    actions=[]
    for item in (result.get('markets',[]) if result.get('status')=='PLANNED' else []):
        actions.append({'market':item['market_id'],'type':'PROPOSE_MARKETING_BUDGET_AND_PRICE',
                        'price_zar':item['price_zar'],'marketing_budget_zar':item['promo_zar'],
                        'require_human_approval':True,'external_execution_enabled':False})
    output.joinpath('pending_approvals.json').write_text(json.dumps({
        'source_digest':source_digest,'status':('AWAITING_HUMAN_APPROVAL' if actions else 'BLOCKED_BY_GOVERNANCE_GATE'),
        'actions':actions,'all_external_execution_disabled':True},indent=2)+'\n')
    if result.get('markets'):
        with output.joinpath('market_plan.csv').open('w',newline='') as f:
            keys=['market_id','market','product_quality','place_reach','price_zar','promo_zar',
                  'forecast_new_paid','forecast_active_paid','weekly_revenue_zar','requests',
                  'contribution_after_marketing_zar']
            writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore')
            writer.writeheader();writer.writerows(result['markets'])
    rows=''.join('<tr>'+''.join(f'<td>{html.escape(str(m.get(k,"")))}</td>' for k in (
            'market','price_zar','promo_zar','forecast_new_paid','forecast_active_paid',
            'requests','weekly_revenue_zar','contribution_after_marketing_zar'))+'</tr>'
            for m in result.get('markets',[]))
    def value(k,default='—'):return html.escape(str(result.get(k,default)))
    metric=result.get('model_metrics',{})
    warnings=''.join('<li>'+html.escape(x)+'</li>' for x in result.get('warnings',[]))
    chartvals=[max(0,m['contribution_after_marketing_zar']) for m in result.get('markets',[])]
    peak=max(chartvals,default=1) or 1
    bars=''.join(f'<div class="chartrow"><span>{html.escape(m["market"])}</span><div class="barbg"><div class="bar" style="width:{max(0,m["contribution_after_marketing_zar"])/peak*100:.1f}%"></div></div><strong>{fmt_zar(m["contribution_after_marketing_zar"])}</strong></div>' for m in result.get('markets',[]))
    output.joinpath('dashboard.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Jarvis X Auto Operator</title>
<style>:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#08121c;color:#e7f2ff;font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}main{max-width:1150px;margin:auto;padding:30px 18px}h1{font-size:clamp(27px,4vw,39px);letter-spacing:-.04em;margin:4px 0}h2{font-size:18px;margin:0 0 12px}.muted{color:#a5b8cb}.eyebrow{color:#79b9f7;text-transform:uppercase;letter-spacing:.14em;font-weight:700;font-size:11px}section{background:#111f2e;border:1px solid #2c4158;border-radius:17px;padding:19px;margin:15px 0}.kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.kpi{border:1px solid #2e4257;border-radius:12px;padding:14px;background:#16273a}.kpi small{color:#b5c6d7;display:block}.kpi strong{font-size:clamp(16px,2.6vw,26px);display:block;margin-top:5px;white-space:nowrap}.pill{display:inline-block;border:1px solid #a68148;border-radius:30px;padding:3px 12px;color:#ffd18b;font-size:11px;background:#312516}.chartrow{display:grid;grid-template-columns:160px 1fr 100px;align-items:center;gap:12px;margin:12px 0}.chartrow strong{text-align:right}.barbg{background:#273b4f;height:16px;border-radius:5px;overflow:hidden}.bar{height:100%;background:linear-gradient(90deg,#3a8bb5,#51d5aa)}table{border-collapse:collapse;width:100%;font-size:12px}th,td{text-align:right;border-bottom:1px solid #294055;padding:9px 6px}th:first-child,td:first-child{text-align:left}th{color:#a9bdcc}ul{padding-left:20px}li{padding-bottom:5px}@media(max-width:790px){.kpis{grid-template-columns:repeat(2,1fr)}.chartrow{grid-template-columns:90px 1fr 70px;gap:4px;font-size:10px}table{font-size:10px;white-space:nowrap}section.overflow{overflow-x:auto}}</style>
<main><div class="eyebrow">Jarvis X / industrial AI commerce / scenario governance</div><h1>Local → Global Auto Operator</h1>
<p class="muted">End-to-end 4P operating plan • ANN forecast • compute allocation • cash reconciliation</p>
<span class="pill">'''+html.escape(result.get('status','UNASSESSED'))+''' · PROPOSALS ONLY · NO LIVE EXECUTION</span>
<section><div class="kpis"><div class="kpi"><small>Forecast new subscriptions</small><strong>'''+value('total_new_paid_forecast')+'''</strong></div><div class="kpi"><small>Weekly revenue</small><strong>'''+fmt_zar(result.get('weekly_revenue_zar',0))+'''</strong></div><div class="kpi"><small>Operating contribution</small><strong>'''+fmt_zar(result.get('weekly_operating_contribution_zar',0))+'''</strong></div><div class="kpi"><small>Compute utilisation</small><strong>'''+format(result.get('compute_utilisation',0)*100,'.1f')+'''%</strong></div></div></section>
<section><h2>Market-by-market contribution</h2>'''+bars+'''</section>
<section class="overflow"><h2>4P forecast and proposed allocation</h2><table><thead><tr><th>Place / market</th><th>Price (R)</th><th>Promotion (R)</th><th>New paid</th><th>Active paid</th><th>Req/week</th><th>Revenue (R)</th><th>Contribution (R)</th></tr></thead><tbody>'''+rows+'''</tbody></table></section>
<section><h2>Cash and model checks</h2><div class="kpis"><div class="kpi"><small>Forecast collected cash</small><strong>'''+fmt_zar(result.get('forecast_customer_collections_zar',0))+'''</strong></div><div class="kpi"><small>Forecast closing cash</small><strong>'''+fmt_zar(result.get('forecast_end_cash_zar',0))+'''</strong></div><div class="kpi"><small>ANN test MAE</small><strong>'''+str(metric.get('test_mae','N/A'))+'''</strong></div><div class="kpi"><small>Naïve test MAE</small><strong>'''+str(metric.get('lag_naive_test_mae','N/A'))+'''</strong></div></div><p class="muted">Test 90% interval coverage: '''+str(metric.get('actual_test_coverage','N/A'))+'''. Financial and compute choices are illustrative; forecast is not a causal marketing experiment.</p></section>
<section><h2>Safety and verification notes</h2><ul>'''+warnings+'''</ul><p class="muted">Input SHA-256: '''+html.escape(source_digest)+'''</p></section></main></html>''',encoding='utf-8')


def run_once(config,source,out):
    if source is None:
        rows=make_demo(config,out)
        mode='SYNTHETIC_DEMO'
    else:
        rows=read_csv(source,config)
        mode='USER_PROVIDED_CSV_UNVERIFIED'
    market_map={m['id']:m for m in config['markets']}
    model,metrics=evaluate(rows,market_map)
    result=plan(config,rows,model,metrics,mode)
    result['run_unix_epoch']=int(time.time())
    digest=hash_data({'config':config,'records':rows})
    write_report(out,result,config,digest)
    with (out/'audit.jsonl').open('a',encoding='utf-8') as log:
        log.write(json.dumps({'event':'PROPOSAL_GENERATED','timestamp':result['run_unix_epoch'],
            'input_sha256':digest,'status':result['status'],
            'external_execution':False,'model_qualified':result['model_qualified']})+'\n')
    print('JARVIS X AUTO OPERATOR |',result['status'],'|',mode)
    print('ANN test MAE: %.2f; last-week naive MAE: %.2f; 90%% interval actual coverage %.1f%%' %
          (metrics['test_mae'],metrics['lag_naive_test_mae'],100*metrics['actual_test_coverage']))
    print('ANN qualified:',result['model_qualified'])
    if result.get('markets'):
        print('Market                 Price   Promotion   New paid  Requests  Contribution')
        for r in result['markets']:
            print(f'{r["market"]:22} {r["price_zar"]:6.0f} {r["promo_zar"]:10,.0f} {r["forecast_new_paid"]:8} {r["requests"]:9} {r["contribution_after_marketing_zar"]:13,.0f}')
        print('Forecast revenue:',fmt_zar(result['weekly_revenue_zar']))
        print('Forecast operating contribution:',fmt_zar(result['weekly_operating_contribution_zar']))
        print('Forecast closing cash:',fmt_zar(result['forecast_end_cash_zar']))
        print('Compute utilised: %.1f%%' % (100*result['compute_utilisation']))
    else:
        print('Reason:',result['reason'])
    if result['status']!='PLANNED':
        print('Expansion/marketing release BLOCKED:',result['status'])
    print('Outputs:',out.resolve())
    print('No external changes made. All actions require approval.')
    return result,digest


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawTextHelpFormatter)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--demo',action='store_true',help='Generate synthetic data and plan using it')
    group.add_argument('--data',type=Path,help='Local CSV with real or other observations')
    parser.add_argument('--config',type=Path,default=HERE/'config.json')
    parser.add_argument('--output',type=Path,default=HERE/'outputs')
    parser.add_argument('--watch-seconds',type=int,default=0,help='Poll for data updates; do NOT execute external actions')
    args=parser.parse_args()
    if args.watch_seconds<0:raise ValueError('Watch interval must be nonnegative')
    last=None
    while True:
        config_bytes=args.config.read_bytes()
        cfg=json.loads(config_bytes)
        if cfg.get('mode')!='simulation': raise ValueError('Only simulation mode is supported; external execution disabled.')
        for m in cfg['markets']:
            if not 0 < m['base_weekly_price_zar'] or not 0 <= m['place_reach'] <= 1:
                raise ValueError('Invalid market price/reach')
        source_info=(args.data.read_bytes() if args.data else b'SYNTHETIC_DEMO')
        fingerprint=hashlib.sha256(source_info+config_bytes).hexdigest()
        if fingerprint!=last:
            run_once(cfg,args.data,args.output)
            last=fingerprint
        elif args.watch_seconds:
            print('No new source input; skipped duplicate plan.')
        if not args.watch_seconds:break
        time.sleep(max(1,args.watch_seconds))


if __name__=='__main__':
    main()