/* Jarvis X / Dr. Moagi: numerical 3D bit kernel + conditioned MLP + inward fold.
 * No external dependencies; CommonJS and browser-global compatible.
 * Scope: small demonstrator, not a foundation model or independent engine swarm.
 */
(function(root,factory){
 const api=factory(); if(typeof module==='object'&&module.exports)module.exports=api;
 else root.DM3D=api;
})(typeof globalThis==='undefined'?this:globalThis,function(){
 'use strict';
 const sigmoid=x=>1/(1+Math.exp(-Math.max(-30,Math.min(30,x))));
 function PRNG(seed){let s=(seed>>>0)||1;return ()=>{s^=s<<13;s^=s>>>17;s^=s<<5;return(s>>>0)/4294967296;};}
 class Runtime {
  constructor(L=16,seed=20261009){
   if(!Number.isSafeInteger(L)||L<8 || (L&(L-1)))throw new Error('side length must be a power of two >=8');
   this.L=L;this.N=L**3;this.random=PRNG(seed);this.λ=.8;this.γ=.2;
   this.neighbors={};for(const d of [1,2,4,8,16,32,64])if(d<L/2)this.neighbors[d]=this.buildOffsets(d);
   this.paths=[];for(let z=0;z<L;z++)for(let y=0;y<L;y++)for(let x=0;x<L;x++)this.paths.push([x,y,z]);
   this.inputs=10;this.hidden=18;
   this.w1=new Float64Array(this.inputs*this.hidden);this.b1=new Float64Array(this.hidden);
   this.w2=new Float64Array(this.hidden);this.b2=0;
   this.initWeights();this.bestValidation=Infinity;this.history=[];this.omega=0;
  }
  id(x,y,z){const m=this.L-1;return ((z&m)*this.L+(y&m))*this.L+(x&m);}
  buildOffsets(d){let arr=new Int32Array(this.N*6);for(let i=0;i<this.N;i++){
   const x=i%this.L, y=Math.floor(i/this.L)%this.L,z=Math.floor(i/(this.L*this.L));let j=6*i;
   arr[j++]=this.id(x+d,y,z);arr[j++]=this.id(x-d,y,z);
   arr[j++]=this.id(x,y+d,z);arr[j++]=this.id(x,y-d,z);
   arr[j++]=this.id(x,y,z+d);arr[j++]=this.id(x,y,z-d);
  }return arr;}
  initWeights(){for(let i=0;i<this.w1.length;i++)this.w1[i]=(this.random()-.5)*.48;
   for(let j=0;j<this.hidden;j++)this.w2[j]=(this.random()-.5)*.45;
   // Preserve strongly predictive local/input features as an initialization shortcut.
   this.w1[0*this.inputs+0]=.9;this.w1[1*this.inputs+1]=.8;
  }
  jumpStep(src,d){const nb=this.neighbors[d]||this.buildOffsets(d),out=new Uint8Array(this.N);
   for(let i=0;i<this.N;i++){let v=src[i];const k=6*i;
    for(let t=0;t<6;t++)v^=src[nb[k+t]];out[i]=v;}
   return out;
  }
  jump(src,iterations){ // Exact K^n over GF(2), periodic on power-of-two cubes.
   if(!(src instanceof Uint8Array)||src.length!==this.N)throw new TypeError('expected a full Uint8Array bit volume');
   if(!Number.isSafeInteger(iterations)||iterations<0)throw new RangeError('iterations must be a nonnegative safe integer');
   let remaining=(Math.trunc(iterations)%(this.L/2)+this.L/2)%(this.L/2);let d=1,r=new Uint8Array(src);
   while(remaining){if(remaining&1)r=this.jumpStep(r,d);remaining=Math.floor(remaining/2);d*=2;}
   return r;
  }
  repeated(src,count){let v=new Uint8Array(src);for(let k=0;k<count;k++)v=this.jumpStep(v,1);return v;}
  smooth(v){const out=new Float64Array(this.N),nb=this.neighbors[1];for(let i=0;i<this.N;i++){
   const k=6*i;out[i]=(v[nb[k]]+v[nb[k+1]]+v[nb[k+2]]+v[nb[k+3]]+v[nb[k+4]]+v[nb[k+5]])/6;
  }return out;}
  volume(shape='sphere',noise=.12){const L=this.L,clean=new Uint8Array(this.N),noisy=new Uint8Array(this.N);
   const sx=(this.random()-.5)*.20,sy=(this.random()-.5)*.20,sz=(this.random()-.5)*.20;
   const scale=.93+.14*this.random();
   for(let i=0;i<this.N;i++){
    let [x,y,z]=this.paths[i];x=(x/L-.5)*2-sx;y=(y/L-.5)*2-sy;z=(z/L-.5)*2-sz;
    let b=0;
    if(shape==='sphere') b=(x*x+y*y+z*z)<.43*scale;
    else if(shape==='torus'){let h=Math.hypot(x,y)-.51;b=h*h+z*z<.115*scale;}
    else if(shape==='double') b= Math.min((x-.32)**2+y*y+z*z,(x+.32)**2+y*y+z*z)<.23*scale;
    else if(shape==='wave') b=Math.abs(z-.28*Math.sin(5*x)*Math.cos(4*y))<.16*scale;
    else throw new Error('unknown shape: '+shape);
    clean[i]=b?1:0;noisy[i]=clean[i]^(this.random()<noise?1:0);
   }return {clean,noisy,shape,noise};
  }
  edge(clean){const nb=this.neighbors[1],out=new Uint8Array(this.N);
   for(let i=0;i<this.N;i++) {let different=0;for(let t=0;t<6;t++)different|=clean[i]!==clean[nb[i*6+t]]?1:0;out[i]=different;}
   return out;
  }
  features(vol,task){const x=vol.noisy,mean=this.smooth(x),jump=this.jump(x,5),out=new Float64Array(this.N*this.inputs);
   for(let i=0;i<this.N;i++){const k=i*this.inputs,[a,b,c]=this.paths[i];
    out[k]=x[i];out[k+1]=mean[i];out[k+2]=Math.abs(x[i]-mean[i]);out[k+3]=jump[i];
    out[k+4]=(a/(this.L-1))*2-1;out[k+5]=(b/(this.L-1))*2-1;out[k+6]=(c/(this.L-1))*2-1;
    out[k+7]=(task==='restore'?1:0);out[k+8]=(task==='edge'?1:0);out[k+9]=(task==='xor'?1:0);
   }return out;
  }
  target(vol,task){if(task==='restore')return vol.clean;if(task==='edge')return this.edge(vol.clean);
   if(task==='xor')return this.jump(vol.noisy,5);throw new Error('unknown task '+task);}
  // Trained MLP: 10 features -> 18 tanh hidden -> sigmoid output.
  forwardRow(f,k,hidden=null){let logit=this.b2;for(let j=0;j<this.hidden;j++){
   let a=this.b1[j];const off=j*this.inputs;for(let m=0;m<this.inputs;m++)a+=this.w1[off+m]*f[k+m];
   const h=Math.tanh(a);if(hidden)hidden[j]=h;logit+=this.w2[j]*h;
  }return sigmoid(logit);}
  predict(features){const pred=new Float64Array(this.N);for(let i=0;i<this.N;i++)pred[i]=this.forwardRow(features,i*this.inputs);return pred;}
  loss(pred,truth){let mse=0,correct=0,tp=0,fp=0,tn=0,fn=0;for(let i=0;i<this.N;i++){
   const delta=pred[i]-truth[i];mse+=delta*delta;
   const p=pred[i]>=.5,t=truth[i]===1;
   if(p===t)correct++;
   if(p&&t)tp++;else if(p)fp++;else if(t)fn++;else tn++;
  }
  const tpr=(tp+fn)?tp/(tp+fn):1,tnr=(tn+fp)?tn/(tn+fp):1;
  return {mse:mse/this.N,accuracy:correct/this.N,balancedAccuracy:.5*(tpr+tnr),iou:(tp+fp+fn)?tp/(tp+fp+fn):1,positiveRate:(tp+fn)/this.N};}
  // Inward AE/AD loop: E=6-neighbor mean, D(z)=(1-gamma) ANN(x)+gamma z.
  // F(z)=(1-lambda)b+lambda*[Q(z)+E(D(z))]/2
  // ||F(z)-F(w)||inf <= lambda*(1+gamma)/2 ||z-w||inf .
  fold(vol,task,opts={}){
   const λ=opts.lambda??this.λ,γ=opts.gamma??this.γ,tol=opts.tol??1e-7,max=opts.maxIter??96;
   if(!(λ>=0&&λ<1&&γ>=0&&γ<=1))throw new Error('contraction parameters invalid');
   const f=this.features(vol,task),h=this.predict(f),x=vol.noisy;
   const avg=this.smooth(x),b=new Float64Array(this.N);
   for(let i=0;i<this.N;i++)b[i]=.45*x[i]+.55*avg[i];
   const eh=this.smooth(h),q=.5*λ*(1+γ),source=new Float64Array(this.N);
   for(let i=0;i<this.N;i++)source[i]=(1-λ)*b[i]+.5*λ*(1-γ)*eh[i];
   let z=new Float64Array(b),bound=Infinity,history=[];const lip=q;
   for(let j=0;j<max;j++){
    const sm=this.smooth(z),next=new Float64Array(this.N);let delta=0;
    for(let i=0;i<this.N;i++){next[i]=source[i]+q*sm[i];delta=Math.max(delta,Math.abs(next[i]-z[i]));}
    z=next;bound=delta/(1-lip);history.push({iteration:j+1,stepDelta:delta,certifiedBound:bound});
    if(bound<=tol)break;
   }
   const predicted=new Float64Array(this.N);for(let i=0;i<this.N;i++)predicted[i]=(1-γ)*h[i]+γ*z[i];
   const target=this.target(vol,task),metrics=this.loss(predicted,target),headMetrics=this.loss(h,target);
   if(opts.recordMemory===true)this.omega=.85*this.omega+.15*metrics.mse;
   return {predicted,head:h,z,b,features:f,target,history,iterations:history.length,
    certifiedBound:bound,contractionLip:lip,metrics,headMetrics,omega:this.omega};
  }
  // Audit whether the XOR ANN can predict without its exact target supplied as feature #3.
  // This is a counterfactual diagnostic, not a retrained no-oracle model.
  auditOracleFreeXor(dataset){
   let count=0,mse=0,accuracy=0;
   for(const d of dataset){if(d.task!=='xor')continue;
    const masked=new Float64Array(d.f);
    for(let i=0;i<this.N;i++)masked[i*this.inputs+3]=0;
    const m=this.loss(this.predict(masked),d.y);
    mse+=m.mse;accuracy+=m.accuracy;count++;
   }
   return {count,mse:count?mse/count:null,accuracy:count?accuracy/count:null};
  }
  createDataset(samples=12){const tasks=['restore','edge','xor'],shapes=['sphere','torus','double','wave'];let dataset=[];
   for(let s=0;s<samples;s++){
    let shape=shapes[s%4],vol=this.volume(shape,.10+.09*this.random());
    for(const task of tasks)dataset.push({f:this.features(vol,task),y:this.target(vol,task),task,shape});
   }return dataset;
  }
  // In-place minibatch SGD, gradients based on binary cross-entropy; training uses ground-truth synthetic fields.
  train(dataset,steps=1200,batch=48,lr=.12){
   let loss=0, samples=0;const hidden=new Float64Array(this.hidden),gw1=new Float64Array(this.w1.length),gb1=new Float64Array(this.hidden),gw2=new Float64Array(this.hidden);
   let gb2=0;
   for(let step=0;step<steps;step++){
    gw1.fill(0);gb1.fill(0);gw2.fill(0);gb2=0;
    const ds=dataset[Math.floor(this.random()*dataset.length)];
    for(let r=0;r<batch;r++){
     const i=Math.floor(this.random()*this.N),k=i*this.inputs,p=this.forwardRow(ds.f,k,hidden),y=ds.y[i];
     let delta=p-y;loss+=(p-y)*(p-y);samples++;
     gb2+=delta;
     for(let j=0;j<this.hidden;j++){
      gw2[j]+=delta*hidden[j];let dh=delta*this.w2[j]*(1-hidden[j]*hidden[j]);gb1[j]+=dh;
      const off=j*this.inputs;for(let m=0;m<this.inputs;m++)gw1[off+m]+=dh*ds.f[k+m];
     }
    }
    const rate=lr/batch;
    for(let i=0;i<this.w1.length;i++)this.w1[i]-=rate*gw1[i];
    for(let j=0;j<this.hidden;j++){this.b1[j]-=rate*gb1[j];this.w2[j]-=rate*gw2[j];}
    this.b2-=rate*gb2;
   }
   const summary={steps,batch,meanSampleMse:loss/samples};this.history.push(summary);return summary;
  }
  evaluate(dataset){let all=0,total=0,acc=0;const byTask={};for(const d of dataset){
    const l=this.loss(this.predict(d.f),d.y);all+=l.mse;acc+=l.accuracy;total++;
    const v=byTask[d.task]||(byTask[d.task]={mse:0,accuracy:0,count:0});v.mse+=l.mse;v.accuracy+=l.accuracy;v.balancedAccuracy=(v.balancedAccuracy||0)+l.balancedAccuracy;v.iou=(v.iou||0)+l.iou;v.positiveRate=(v.positiveRate||0)+l.positiveRate;v.count++;
   }for(const key of Object.keys(byTask)){const v=byTask[key];v.mse/=v.count;v.accuracy/=v.count;v.balancedAccuracy/=v.count;v.iou/=v.count;v.positiveRate/=v.count;}
   return {mse:all/total,accuracy:acc/total,byTask};
  }
 }
 return {Runtime,PRNG};
});
