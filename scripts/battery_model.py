"""Battery dispatch with one common, strictly convex action tie break.

All powers are kW and state energies kWh. Positive net is import. Forecasts
change settlement, never the unconstrained battery feasible set. Clarabel is
used for deterministic replay; decision_layer.py supplies the native polished
KKT differentiation used in the reported experiments. The optional legacy
CVXPYlayers constructor below is not used by those experiments.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import sparse
import clarabel


@dataclass(frozen=True)
class Battery:
    capacity: float = 10.0
    power: float = 3.0
    efficiency: float = .95
    dt: float = .5
    throughput: float = .02
    epsilon: float = 1e-4
    initial_fraction: float = .5
    minimum_fraction: float = .1
    maximum_fraction: float = .9


def tariff(kind='tou', n=48):
    t=np.arange(n)*.5 % 24
    if kind=='tou': buy=np.where((t<7)|(t>=22),.18,np.where((t>=16)&(t<21),.5,.3))
    elif kind=='flat': buy=np.full(n,.3)
    elif kind=='equal': buy=np.where((t<7)|(t>=22),.18,np.where((t>=16)&(t<21),.5,.3))
    else: raise ValueError(kind)
    return buy, buy.copy() if kind=='equal' else np.full(n,.08)


def realized_bill(net,charge,discharge,buy,sell,battery=Battery()):
    g=np.asarray(net)+charge-discharge
    return battery.dt*np.sum(buy*np.maximum(g,0)+sell*np.minimum(g,0)+battery.throughput*(charge+discharge),axis=-1)


class DispatchQP:
    """Reusable direct sparse QP, also allowing shared-action scenario paths.

    Scenario net shape is [scenario,time]; all paths share charge and discharge.
    No pathwise perfect foresight is introduced. We omit the forecast-only
    sell*net constant when solving but include it in realized settlement.
    """
    def __init__(self,battery=Battery(),kind='tou',n=48,n_scenarios=1,initial=None,terminal=None,tolerance=1e-8,perturbation=None):
        self.battery=battery;self.n=n;self.n_scenarios=n_scenarios
        b=battery;self.buy,self.sell=tariff(kind,n)
        self.equal=bool(np.all(self.buy==self.sell))
        ns=0 if self.equal else n_scenarios
        self.dim=3*n+ns*n
        eye=sparse.eye(n,format='csc');zero=sparse.csc_matrix((n,n))
        eg=eye-sparse.diags(np.ones(n-1),-1,shape=(n,n),format='csc')
        energy=sparse.hstack([-b.efficiency*b.dt*eye,b.dt/b.efficiency*eye,eg,sparse.csc_matrix((n,ns*n))],format='csc')
        end=sparse.csc_matrix(([1.],([0],[3*n-1])),shape=(1,self.dim))
        eq=sparse.vstack([energy,end],format='csc')
        initial=b.initial_fraction*b.capacity if initial is None else initial
        terminal=initial if terminal is None else terminal
        rhs=np.zeros(n+1);rhs[0]=initial;rhs[-1]=terminal
        blocks=[eq];rights=[rhs];self.net_row=n+1
        if not self.equal:
            grid=sparse.hstack([sparse.vstack([eye]*ns),sparse.vstack([-eye]*ns),sparse.csc_matrix((ns*n,n)),-sparse.eye(ns*n)],format='csc')
            blocks.append(grid);rights.append(np.zeros(ns*n))
        # Only non-negative charging/discharging/import; energy has explicit bounds.
        neg=sparse.hstack([-sparse.eye(2*n),sparse.csc_matrix((2*n,n+ns*n))],format='csc')
        upper=-neg
        emin=sparse.hstack([sparse.csc_matrix((n,2*n)),-eye,sparse.csc_matrix((n,ns*n))],format='csc')
        emax=-emin
        blocks.extend([neg,upper,emin,emax]);rights.extend([np.zeros(2*n),np.full(2*n,b.power),np.full(n,-b.minimum_fraction*b.capacity),np.full(n,b.maximum_fraction*b.capacity)])
        if ns:
            umin=sparse.hstack([sparse.csc_matrix((ns*n,3*n)),-sparse.eye(ns*n)],format='csc')
            blocks.append(umin);rights.append(np.zeros(ns*n))
        self.A=sparse.vstack(blocks,format='csc');self.rhs=np.concatenate(rights)
        p=np.r_[np.full(2*n,2*b.epsilon/b.power**2),np.full(n,2*b.epsilon/b.capacity**2),np.zeros(ns*n)]
        self.P=sparse.diags(p,format='csc')
        self.q=np.r_[b.dt*(b.throughput+self.sell),b.dt*(b.throughput-self.sell),np.zeros(n),np.tile(b.dt*(self.buy-self.sell)/max(1,ns),ns)]
        if perturbation is not None:
            self.q[:2*n]+=np.asarray(perturbation)
        settings=clarabel.DefaultSettings();settings.verbose=False
        settings.tol_gap_abs=tolerance;settings.tol_gap_rel=tolerance;settings.tol_feas=tolerance
        settings.max_iter=200
        self.solver=clarabel.DefaultSolver(self.P,self.q,self.A,self.rhs,[clarabel.ZeroConeT(n+1),clarabel.NonnegativeConeT(self.A.shape[0]-n-1)],settings)
    def solve(self,net,return_diagnostics=False):
        net=np.asarray(net,dtype=float)
        if not np.all(np.isfinite(net)): raise ValueError('Nonfinite net forecast')
        if not self.equal:
            if net.size!=self.n*self.n_scenarios: raise ValueError(net.shape)
            self.rhs[self.net_row:self.net_row+net.size]=-net.ravel()
            self.solver.update(b=self.rhs)
        res=self.solver.solve()
        if str(res.status) not in ('Solved','AlmostSolved'): raise RuntimeError(f'QP {res.status}')
        x=np.array(res.x);n=self.n
        c=x[:n];d=x[n:2*n];e=x[2*n:3*n]
        if min(c.min(),d.min()) < -1e-5 or np.max(np.minimum(c,d))>1e-4: raise RuntimeError('Invalid simultaneous or negative battery action')
        if return_diagnostics:
            residual=self.A@x-self.rhs
            return c,d,e,{'status':str(res.status),'iterations':res.iterations,'primal_residual':res.r_prim,'dual_residual':res.r_dual,'energy_balance_max':float(abs(residual[:n+1]).max()),'max_simultaneous_kw':float(np.minimum(c,d).max()),'constraint_violation':float(max(0,residual[n+1:].max()))}
        return c,d,e


def make_differentiable_layer(battery=Battery(),kind='tou',n=48):
    import cvxpy as cp
    from cvxpylayers.torch import CvxpyLayer
    b=battery;buy,sell=tariff(kind,n)
    net=cp.Parameter(n)
    c=cp.Variable(n);d=cp.Variable(n);e=cp.Variable(n)
    constraints=[c>=0,d>=0,c<=b.power,d<=b.power,e>=b.minimum_fraction*b.capacity,e<=b.maximum_fraction*b.capacity,e[0]==b.initial_fraction*b.capacity+b.dt*(b.efficiency*c[0]-d[0]/b.efficiency),e[1:]==e[:-1]+b.dt*(b.efficiency*c[1:]-d[1:]/b.efficiency),e[-1]==b.initial_fraction*b.capacity]
    g=net+c-d
    obj=b.dt*(sell@(c-d)+b.throughput*cp.sum(c+d))+b.epsilon*(cp.sum_squares(c/b.power)+cp.sum_squares(d/b.power)+cp.sum_squares(e/b.capacity))
    if not np.all(buy==sell):
        u=cp.Variable(n);constraints.extend([u>=g,u>=0]);obj+=b.dt*((buy-sell)@u)
    else:
        # net enters a pure constant only. Keep it as a parameter for the zero-gradient audit.
        obj+=b.dt*(sell@net)
    problem=cp.Problem(cp.Minimize(obj),constraints)
    assert problem.is_dpp()
    return CvxpyLayer(problem,parameters=[net],variables=[c,d,e],solver=cp.CLARABEL)


def self_consumption(net,battery=Battery()):
    """Causal physical rule; no artificial daily terminal reset."""
    b=battery;arr=np.asarray(net)
    if arr.ndim>2: raise ValueError('Call self_consumption separately for each household; input is slots or days x slots')
    flat=arr.reshape(-1);c=np.zeros_like(flat);d=np.zeros_like(flat)
    energy=b.initial_fraction*b.capacity
    for t,p in enumerate(flat):
        if p<0: c[t]=min(-p,b.power,(b.maximum_fraction*b.capacity-energy)/(b.dt*b.efficiency))
        else: d[t]=min(p,b.power,(energy-b.minimum_fraction*b.capacity)*b.efficiency/b.dt)
        energy+=b.dt*(b.efficiency*c[t]-d[t]/b.efficiency)
    return c.reshape(arr.shape),d.reshape(arr.shape),energy
