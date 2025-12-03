import abc
import torch
import torch.nn.functional as F
from torch import nn
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, List
# from .tuples import SessionBatch, SessionOutput
from collections import defaultdict
from itertools import combinations

class AbstractLoss(ABC):
    @abstractmethod
    def compute_loss(
        self,
        model,
        output,
        batch,
        hps,
        device,
    ):
        """
        Compute the loss given the model output and any extra required data.

        Returns:
        - loss: torch.Tensor
        - metrics: Dict[str, torch.Tensor]
        """
        pass
    def get_hyperparameters(self) -> Any:
        pass

class LossStack:
    def __init__(self, loss_list: List[AbstractLoss] = []):
        self.loss_list = loss_list

    def compute_losses(
        self,
        model,
        output,
        batch,
        hps,
        device,
    ):
        total_loss = 0.0
        metrics = {}
        if len(self.loss_list) == 0:
            return total_loss, metrics
        for loss_fn in self.loss_list:
            loss, loss_metrics = loss_fn.compute_loss(model, output, batch, hps, device)
            total_loss += loss
            metrics.update(loss_metrics)
        return total_loss, metrics
    
    def get_hyperparameters(self) -> Any:
        # If loss_list is empty, return an empty dictionary
        if len(self.loss_list) == 0:
            return {}
        else:
            return {loss_fn.__class__.__name__: loss_fn.get_hyperparameters() for loss_fn in self.loss_list}
        
class LocalJacobianMinSingularLoss(AbstractLoss):
    """
    Penalize small singular values of J = ∂rates/∂z so that every
    latent dimension has a non-trivial influence on the rate space.

    For each area:
      - sample latent points z
      - construct J by JVPs across chosen latent cols
      - compute singular values per sample
      - penalize values below a threshold (min_sval_target)
    """

    def __init__(
        self,
        minSV_scale: float,
        sample_size: int = 5000,
        col_sample_size: int = 0,        # same semantics as your ISO loss
        neuron_sample_size: int = 0,
        min_sval_target: float = 0.02,   # "no dead-dim" threshold
        softplus_beta: float = 20.0,     # sharpness of below-target penalty
        create_graph: bool = True,
        max_cols_cap: int = 64,
        apply_recon_means: bool = True,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.minSV_scale = minSV_scale
        self.sample_size = sample_size
        self.col_sample_size = col_sample_size
        self.neuron_sample_size = neuron_sample_size
        self.min_sval_target = min_sval_target
        self.softplus_beta = softplus_beta
        self.create_graph = create_graph
        self.max_cols_cap = max_cols_cap
        self.apply_recon_means = apply_recon_means
        self.eps = eps

    def get_hyperparameters(self):
        return {
            "minSV_scale": self.minSV_scale,
            "sample_size": self.sample_size,
            "col_sample_size": self.col_sample_size,
            "neuron_sample_size": self.neuron_sample_size,
            "min_sval_target": self.min_sval_target,
            "softplus_beta": self.softplus_beta,
            "create_graph": self.create_graph,
            "max_cols_cap": self.max_cols_cap,
            "apply_recon_means": self.apply_recon_means,
            "eps": self.eps,
        }

    # ----- utility: identical to your ISO loss -----
    @staticmethod
    def _as_B1D(z): return z.view(z.shape[0],1,z.shape[1])

    @staticmethod
    def _pick_tensor(x):
        if torch.is_tensor(x): return x
        if isinstance(x, dict):
            for k in ("rate","rates","mean","means","log_rate","pre_rate","loc"):
                if k in x and torch.is_tensor(x[k]): return x[k]
            for v in x.values():
                if torch.is_tensor(v): return v
        if isinstance(x, (tuple, list)):
            for v in x:
                if torch.is_tensor(v): return v
        raise RuntimeError("readout returned no tensor.")

    def _rates_from_z(self, model, z_flat):
        Z = self._as_B1D(z_flat)
        outp = model.readout[0](Z) # need to fix this for multi-session, right now takes the readout of the first session
        params = self._pick_tensor(outp)
        if self.apply_recon_means:
            out = model.recon[0].compute_means(params.transpose(1,2))
        else:
            out = params
        if out.dim()==3 and out.size(1)==1:
            out = out[:,0,:]
        return out.view(out.shape[0], -1)
    
    def _choose_columns(self, D, device):
        if self.col_sample_size > 0:
            K = min(self.col_sample_size, D)
            return torch.randperm(D, device=device)[:K]
        K = min(D, self.max_cols_cap)
        if K < D:
            return torch.randperm(D, device=device)[:K]
        return torch.arange(D, device=device)

    # ----- core singular-value penalty -----

    def _min_singular_term(self, model, z):
        S, D = z.shape
        device = z.device

        # closure for JVP
        def F(inp): return self._rates_from_z(model, inp)

        z_req = z.clone().detach().requires_grad_(True)
        rates0 = F(z_req)

        # neuron subsampling
        if self.neuron_sample_size > 0 and self.neuron_sample_size < rates0.shape[1]:
            idxn = torch.randperm(rates0.shape[1], device=device)[:self.neuron_sample_size]
        else:
            idxn = None
    
        # pick latent dims to differentiate wrt
        idx_cols = self._choose_columns(D, device)
        K = idx_cols.numel()

        J_cols = []
        for d in idx_cols.tolist():
            v = torch.zeros_like(z_req)
            v[:, d] = 1.0
            y0, jvp = torch.autograd.functional.jvp(F, z_req, v, create_graph=self.create_graph)
            if idxn is not None:
                jvp = jvp[:, idxn]
            J_cols.append(jvp)

        # J: [S, N_sub, K]
        J = torch.stack(J_cols, dim=-1)

        # singular values per sample
        # we reshape each sample as [N_sub, K] and run SVD
        svals_list = []
        for sidx in range(S):
            Js = J[sidx]   # [N_sub, K]
            # SVD is stable because N_sub >= K in usual models
            # using full_matrices=False keeps it cheap
            try:
                sv = torch.linalg.svdvals(Js)  # [min(N_sub,K)]
            except:
                # fallback: CPU
                sv = torch.linalg.svdvals(Js.cpu()).to(device)
            svals_list.append(sv)
        # pack to [S, K] (pad small batches)
        max_k = max(len(v) for v in svals_list)
        svals = torch.stack([
            torch.nn.functional.pad(v, (0, max_k - len(v)), value=0.0)
            for v in svals_list
        ], dim=0)

        # --- penalty: push up low singular values ---
        # smooth hinge via softplus
        # hinge = max(0, target - sigma)
        hinge = torch.clamp(self.min_sval_target - svals, min=0.0)
        penalty = torch.nn.functional.softplus(self.softplus_beta * hinge).mean()

        metrics = {
            "min_sval_mean": svals.mean().detach(),
            "min_sval_min": svals.min().detach(),
            "min_sval_penalty": penalty.detach(),
            "min_sval_target": torch.as_tensor(self.min_sval_target, device=device),
            "jac_cols": torch.as_tensor(float(K), device=device),
        }

        return penalty, metrics
    
    # ----- main hook -----

    def compute_loss(self, model, output, batch, hps, device):
        sessions = sorted(output.keys())
        # area_names = model.area_names

        total = torch.tensor(0.0, device=device)
        metrics = {}

        # get all latents across sessions → flatten → subsample
        Z_list = [output[s].factors.to(device) for s in sessions]
        Z = torch.cat(Z_list, dim=0)      # [B_all, T, D]
        B,T,D = Z.shape
        z_all = Z.reshape(B*T, D)

        if z_all.shape[0] > self.sample_size:
            idx = torch.randperm(z_all.shape[0], device=device)[:self.sample_size]
            z = z_all[idx]
        else:
            z = z_all

        term, m = self._min_singular_term(model, z)
        total = total + term

        metrics.update({
            f"minsv_term": term.detach(),
            f"minsv_mean": m["min_sval_mean"],
            f"minsv_min": m["min_sval_min"],
            f"minsv_target": m["min_sval_target"],
            f"minsv_cols": m["jac_cols"],
        })

        loss = self.minSV_scale * total
        metrics.update({
            "minsv_loss": loss.detach(),
            "minsv_total": total.detach(),
            "minsv_scale": torch.as_tensor(self.minSV_scale, device=device),
        })
        return loss, metrics
