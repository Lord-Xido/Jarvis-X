import {run} from '../engine.mjs';
const cases=[
 {text:'jaw swelling and trouble swallowing',flags:[],language:'en',expect:'SIMULATION_RED_FLAG'},
 {text:'other symptom',flags:[],language:'zu',expect:'UNASSESSED_NOT_CLEARED'},
 {text:'difficulty breathing',flags:[],language:'st',expect:'SIMULATION_RED_FLAG'},
 {text:'demo text',flags:['synthetic_red_flag'],language:'af',expect:'SIMULATION_RED_FLAG'},
 {text:'rash',flags:[],language:'en',expect:'UNASSESSED_NOT_CLEARED'},
 {text:'routine synthetic scenario',flags:[],language:'en',expect:'UNASSESSED_NOT_CLEARED'}
];
let ok=0,handoffs=0;
for(let i=0;i<120;i++){
 const c=cases[i%cases.length],r=run(c);
 if(r.classification!==c.expect||r.stageProofs.length!==3||r.orders.length!==0)
   throw Error('Simulation failed '+i);
 ok++;handoffs+=r.stageProofs.length;
}
console.log(JSON.stringify({simulation:'synthetic',cases:ok,authenticatedHandoffs:handoffs,status:'PASS'}));
