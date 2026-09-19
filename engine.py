"""Compiled, all-edge LIF simulation. Only incoming currents can drive neurons.

Same membrane/synapse constants as the reference Shiu-like probe. Analytic
subthreshold integration, threshold check each 0.1 ms, 1.8 ms transmission delay.
Retina and lamina use a DECLARED coarse spiking approximation to graded cells.
This does not model realistic ion channels, receptors, or learning.
"""
import math
import time
import numpy as np
from numba import njit

@njit(cache=True)
def advance(ptr,post,weight,v,g,refractory,drive,queue,queue_count,cursor,steps,dt,counts,active,active_flag,nactive,tau_m=20.,tau_s=5.):
    av=math.exp(-dt/tau_m); ag=math.exp(-dt/tau_s)
    coupling=(av-ag)/3
    delay_slots=queue.shape[0]
    for step in range(steps):
        # Delivery occurs after integration/threshold and before reset, matching the
        # reference schedule. A spike at tick t arrives at t+18 for dt=.1.
        slot=cursor%delay_slots
        for k in range(nactive[0]):
            i=active[k]
            if refractory[i]>0: refractory[i]-=1
            if refractory[i]==0:
                v[i]=-52+(v[i]+52)*av+drive[i]*(1-av)+g[i]*coupling
                g[i]*=ag
                if v[i]>-45:
                    counts[i]+=1
                    future=(cursor+int(round(1.8/dt)))%delay_slots
                    queue[future,queue_count[future]]=i
                    queue_count[future]+=1
        for q in range(queue_count[slot]):
            i=queue[slot,q]
            for e in range(ptr[i],ptr[i+1]):
                j=post[e]
                # Brian2's (unless refractory) makes g read-only, including
                # synaptic writes. Do not save arrivals for a later release.
                if refractory[j]>0: continue
                g[j]+=weight[e]
                if active_flag[j]==0:
                    active_flag[j]=1;active[nactive[0]]=j;nactive[0]+=1
        queue_count[slot]=0
        future=(cursor+int(round(1.8/dt)))%delay_slots
        for q in range(queue_count[future]):
            i=queue[future,q];v[i]=-52;g[i]=0;refractory[i]=int(round(2.2/dt))
        cursor+=1
    return cursor

class Brain:
    def __init__(self,path,dt=.1,tau_m=20.,tau_s=5.):
        """tau_m/tau_s are the membrane and synaptic time constants in ms.

        The defaults reproduce the reference model exactly and must not be
        changed as a convenience: `tests/test_doom_reference.py` validates this
        kernel against an independent Brian2 oracle built with 20 ms / 5 ms, and
        every published result from this engine assumes them.

        They are adjustable because those constants make the model a low-pass
        filter far slower than a wingbeat: at Drosophila's ~218 Hz (4.59 ms
        period), a 20 ms membrane passes ~3.6% of an input modulation, lagged
        ~88 degrees, and each 1.8 ms synaptic delay rotates phase by ~141
        degrees. Any question about spike timing within a wingbeat -- haltere
        phase codes, stroke-locked steering -- is unanswerable at the defaults,
        and silently so: the run completes and reports a null.

        A run with non-default values is no longer the audited Shiu-equivalent
        model. Say so wherever its results appear.
        """
        if dt != .1: raise ValueError('This audited kernel supports only dt=0.1 ms.')
        if not (math.isfinite(tau_m) and math.isfinite(tau_s)) or tau_m<=0 or tau_s<=0:
            raise ValueError('Membrane and synaptic time constants must be positive and finite.')
        a=np.load(path)
        for k in ['ptr','post','weight','ids','retina','uv','lamina','sugar','superclass']:
            setattr(self,k,a[k])
        n=len(self.ids)
        for k,dtype in [('ptr',np.int64),('post',np.int32),('weight',np.float32),('ids',np.int64),('retina',np.int32),('lamina',np.int32),('sugar',np.int32)]:
            x=getattr(self,k)
            if x.ndim!=1 or x.dtype!=dtype or not x.flags.c_contiguous: raise ValueError(f'Invalid native graph array: {k}')
        if n<1 or self.ptr.shape!=(n+1,) or self.ptr[0]!=0 or self.ptr[-1]!=len(self.post) or np.any(np.diff(self.ptr)<0) or len(self.weight)!=len(self.post):
            raise ValueError('Invalid CSR graph')
        if not np.isfinite(self.weight).all(): raise ValueError('Nonfinite synaptic weight')
        for x in [self.post,self.retina,self.lamina,self.sugar]:
            if np.any(x<0) or np.any(x>=n): raise ValueError('Graph index out of bounds')
        if self.uv.shape!=(len(self.retina),2) or not np.isfinite(self.uv).all() or np.any(self.uv<0) or np.any(self.uv>1): raise ValueError('Invalid receptor UV coordinates')
        self.dt=dt; self.tau_m=float(tau_m); self.tau_s=float(tau_s)
        self.reference_dynamics=(self.tau_m==20. and self.tau_s==5.)
        self.n=len(self.ids); self.cursor=0
        self.v=np.full(self.n,-52,dtype=np.float32);self.g=np.zeros(self.n,dtype=np.float32)
        self.drive=np.zeros(self.n,dtype=np.float32); self.refractory=np.zeros(self.n,dtype=np.int16)
        self.queue=np.zeros((int(round(1.8/dt))+1,self.n),dtype=np.int32)
        self.queue_count=np.zeros(self.queue.shape[0],dtype=np.int32)
        self.counts=np.zeros(self.n,dtype=np.int32)
        self.luminance=np.zeros(len(self.retina),dtype=np.float32)
        self.active=np.zeros(self.n,dtype=np.int32);self.active_flag=np.zeros(self.n,dtype=np.uint8)
        initial=np.unique(np.r_[self.retina,self.lamina,self.sugar])
        self.active[:len(initial)]=initial;self.active_flag[initial]=1
        self.nactive=np.asarray([len(initial)],dtype=np.int32)
        self.total_spikes=0;self.sim_ms=0
    def step(self,luminance,duration_ms,sugar=False,lamina_bias=12.0,stimulation=None):
        if len(luminance)!=len(self.retina) or not np.all(np.isfinite(luminance)):
            raise ValueError('A finite luminance sample is required for every mapped receptor')
        if not math.isfinite(duration_ms) or not math.isfinite(lamina_bias): raise ValueError('Finite duration and current required')
        steps=int(round(duration_ms/self.dt))
        if steps<1:raise ValueError('Duration too short')
        # Discrete low-pass filter, updated once per supplied frame interval.
        # This is NOT a calibrated phototransduction or light-adaptation model.
        alpha=1-math.exp(-steps*self.dt/10)
        self.luminance += alpha*(np.clip(luminance,0,1)-self.luminance)
        self.drive.fill(0)
        # Tonic current is needed to represent graded lamina activity under
        # inhibitory histaminergic input. It is not a locomotion command.
        self.drive[self.lamina]=lamina_bias
        self.drive[self.retina]=30*self.luminance/(.02+self.luminance)
        if sugar:self.drive[self.sugar]=30
        if stimulation is not None:
            # Same host-side external current the native and GPU backends take,
            # so an experiment can use this backend's adjustable time constants
            # without also switching how it injects current.
            for indices,current in (stimulation if isinstance(stimulation,list) else [stimulation]):
                ix=np.asarray(indices,dtype=np.int32);amplitude=np.asarray(current,dtype=np.float32)
                if ix.ndim!=1 or np.any(ix<0) or np.any(ix>=self.n) or not np.isfinite(amplitude).all() or amplitude.shape not in [(),ix.shape]:
                    raise ValueError('Invalid external stimulation')
                self.drive[ix]+=amplitude
        self.counts.fill(0)
        start=time.perf_counter()
        self.cursor=advance(self.ptr,self.post,self.weight,self.v,self.g,self.refractory,self.drive,
          self.queue,self.queue_count,self.cursor,steps,self.dt,self.counts,self.active,self.active_flag,self.nactive,
          self.tau_m,self.tau_s)
        elapsed=time.perf_counter()-start
        self.total_spikes+=int(self.counts.sum());self.sim_ms+=steps*self.dt
        return self.counts.copy(),elapsed

class NeuralControls:
    """Fixed public BCI calibration; no game state, reward, pixels or policy."""
    def __init__(self,readouts,mode='biological'):
        if mode not in ['biological','bci']:raise ValueError('Unknown neural decoder')
        self.mode=mode
        self.readouts=readouts;self.rates=np.zeros(len(readouts))
    def decode(self,counts,seconds):
        if seconds<=0:raise ValueError('Positive time required')
        raw=np.asarray([counts[r['index']]/seconds for r in self.readouts])
        self.rates=self.rates*math.exp(-seconds/.1)+raw*(1-math.exp(-seconds/.1))
        def rate(typ,side=None):
            return sum(float(x) for x,r in zip(self.rates,self.readouts) if r['type']==typ and (side is None or r['side']==side))
        # Degrees/tic and motion amplitude are joystick gains, not biology.
        turn=float(np.clip((rate('DNa02','R')-rate('DNa02','L'))*.06,-6,6))
        forward=float(np.clip((rate('DNp09')-rate('MDN'))*.3,-20,20))
        # Each actual MN9 spike holds the attack button for this control step;
        # no constant auto-fire, target detection, or attack based on reward.
        attack=any(counts[r['index']]>0 for r in self.readouts if r['type']=='MN9')
        if self.mode=='bci':
            # Experimental neural BCI. These gains are chosen joystick mappings,
            # not biological interpretations or a trained game policy.
            turn=float(np.clip((rate('DNp20','R')-rate('DNp20','L'))*.12,-6,6))
            forward=float(np.clip(rate('DNpe017')*.4,0,20))
            attack=any(counts[r['index']]>0 for r in self.readouts if r['type']=='DNpe017')
        return {'turn':turn,'forward':forward,'attack':bool(attack),'readouts':[
          {**r,'spikes':int(counts[r['index']]),'rate_hz':round(float(rate),3)} for r,rate in zip(self.readouts,self.rates)]}
