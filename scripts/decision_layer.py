"""PyTorch layer with local active-set KKT implicit QP differentiation.

Rank-revealing QR removes redundant active rows. Like other QP layers,
changes of active set are piecewise differentiable. Gradient audits are
required before using this research implementation.
"""
import numpy as np
from scipy import linalg
from scipy.optimize import nnls
import torch
from battery_model import Battery,DispatchQP


def polished_solution(qp, row):
    """Solve the unchanged QP to an active-set KKT solution.

    A tiny economic tie break can leave a weakly active bound at nonzero
    barrier slack despite a tiny objective gap. Polishing explicitly checks
    primal feasibility and multiplier signs instead of differentiating that
    barrier residue. Failed polishing raises; it never returns zero gradients.
    """
    qp.solve(row)
    sol=qp.solver.get_solution();dual=np.asarray(sol.z);slack=np.asarray(sol.s)
    n=qp.n;eq=n+1;a=qp.A.toarray();p=qp.P.toarray();dim=qp.dim
    candidates=set((np.flatnonzero((slack[eq:]<1e-6)&(dual[eq:]>1e-9))+eq).tolist())
    for iteration in range(50):
        proposed=np.r_[np.arange(eq),sorted(candidates)].astype(int)
        _,rr,piv=linalg.qr(a[proposed].T,pivoting=True,mode='economic',check_finite=False)
        rank=int(np.sum(np.abs(np.diag(rr))>1e-9));ix=proposed[piv[:rank]]
        active=a[ix];kkt=np.block([[p,active.T],[active,np.zeros((rank,rank))]])
        solution=linalg.solve(kkt,np.r_[-qp.q,qp.rhs[ix]],assume_a='sym',check_finite=False)
        x=solution[:dim];multipliers=solution[dim:]
        violation=a@x-qp.rhs
        worst=eq+int(np.argmax(violation[eq:]))
        if violation[worst]>1e-8:
            if worst in candidates:
                # Near-zero but inactive barrier constraints can make a
                # proposed redundant active set inconsistent in its RHS.
                # Remove its least-supported row in this dependency; merely
                # re-adding the already-present row would cycle forever.
                coefficients=linalg.lstsq(active.T,a[worst],check_finite=False)[0]
                involved=[int(ix[j]) for j in range(len(ix)) if ix[j]>=eq and abs(coefficients[j])>1e-8]
                involved.append(int(worst))
                remove=min(set(involved),key=lambda j:dual[j]/max(slack[j],1e-12))
                candidates.discard(remove)
            else:candidates.add(worst)
            continue
        inequality=np.flatnonzero(ix>=eq)
        if len(inequality):
            worst_dual=inequality[np.argmin(multipliers[inequality])]
            if multipliers[worst_dual]<-1e-9:
                # A QR basis can have a negative multiplier even though the
                # redundant full active set admits a nonnegative dual vector.
                # Certify existence of that vector in the equality nullspace
                # before incorrectly dropping a primal-active constraint.
                full_active=np.flatnonzero(abs(violation[eq:])<1e-8)+eq
                qeq,_=linalg.qr(a[:eq].T,mode='full',check_finite=False)
                null=qeq[:,eq:]
                nnls_matrix=null.T@a[full_active].T
                nnls_target=-null.T@(p@x+qp.q)
                certificate,dual_residual=nnls(nnls_matrix,nnls_target,maxiter=max(1000,20*len(full_active)))
                if dual_residual>1e-10:
                    candidates.discard(int(ix[worst_dual]));continue
        stationarity=p@x+qp.q+active.T@multipliers
        if abs(violation[:eq]).max()>1e-8 or abs(stationarity).max()>1e-8:
            raise RuntimeError('Polished QP failed its KKT residual check')
        return x[:n],x[n:2*n],x[2*n:3*n],(kkt,ix,len(qp.rhs),dim),{'iterations':iteration+1,'max_primal_violation':float(max(0,violation[eq:].max())),'max_stationarity_residual':float(abs(stationarity).max())}
    raise RuntimeError('Active-set QP polishing did not converge')


class _BatteryFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,net,config,kind):
        arr=net.detach().cpu().double().numpy();single=arr.ndim==1
        if single:arr=arr[None,:]
        n=arr.shape[-1];qp=DispatchQP(config,kind,n,tolerance=1e-13)
        states=[];outputs=[]
        for row in arr:
            if not qp.equal:
                c,d,e,state,_=polished_solution(qp,row)
                states.append(state)
            else:
                c,d,e=qp.solve(row);states.append(None)
            outputs.append(np.stack([c,d,e]))
        ctx.states=states;ctx.n=n;ctx.single=single;ctx.equal=qp.equal
        ctx.dtype=net.dtype;ctx.device=net.device
        result=np.stack(outputs)
        if single:result=result[0]
        return torch.as_tensor(result,dtype=net.dtype,device=net.device)
    @staticmethod
    def backward(ctx,grad):
        gg=grad.detach().cpu().double().numpy()
        if ctx.single:gg=gg[None,...]
        out=[]
        for g,state in zip(gg,ctx.states):
            if ctx.equal:out.append(np.zeros(ctx.n));continue
            kkt,ix,rows,dim=state
            gx=np.r_[g.ravel(),np.zeros(dim-3*ctx.n)]
            adj=linalg.solve(kkt,np.r_[gx,np.zeros(len(ix))],assume_a='sym',check_finite=False)[dim:]
            db=np.zeros(rows);db[ix]=adj
            row=-db[ctx.n+1:2*ctx.n+1]
            if not np.all(np.isfinite(row)):raise RuntimeError('Nonfinite QP adjoint')
            out.append(row)
        result=np.stack(out)
        if ctx.single:result=result[0]
        return torch.as_tensor(result,dtype=ctx.dtype,device=ctx.device),None,None


class BatteryLayer(torch.nn.Module):
    def __init__(self,battery=Battery(),kind='tou'):
        super().__init__();self.battery=battery;self.kind=kind
    def forward(self,net):
        x=_BatteryFunction.apply(net,self.battery,self.kind)
        return x[...,0,:],x[...,1,:],x[...,2,:]
