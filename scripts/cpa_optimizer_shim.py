"""Put CPA's chemical projection back into the optimizer — an upstream defect, fixed in OUR code.

THE DEFECT. cpa-tools 0.8.8 builds, when use_rdkit_embeddings=True, a FROZEN molecular embedding
plus a trainable nn.Linear that maps it into the latent space:

    cpa/_utils.py  PerturbationNetwork.__init__
        self.pert_embedding      = drug_embeddings          # frozen, the molecular vector
        self.pert_transformation = nn.Linear(...)           # the chemical projection
    cpa/_utils.py  PerturbationNetwork.forward
        drug_embeddings = self.pert_transformation(drug_embeddings...)

but `configure_optimizers` collects only encoder, decoder, pert_network.pert_embedding,
covars_embeddings, pert_network.dosers and the adversaries (cpa/_task.py:365-396).
`pert_transformation` is in NO optimizer group. Verified on a real module, not by reading:

    PerturbationNetwork(n_perts=7, n_latent=8, drug_embeddings=nn.Embedding(7, 16))
    -> pert_transformation.{weight,bias}: requires_grad=True, in ae_params? False, in doser_params? False

So it receives gradients and never a step: the chemical projection stays at its random
initialisation for the whole run, and chemistry reaches the decoder only through a fixed random
linear map. That is not chemCPA's published behaviour -- the separate chemCPA implementation
(theislab/chemCPA, lightning_module.py) does put its chemical encoder parameters in the optimizer.
It is an upstream bug in the release we run, not a limit of the method, and the one deposited cell
it produced is CPA x T5u (execution_model chemCPA, pearson_delta 0.1116).

THE FIX, AND WHY IT IS SHAPED THIS WAY. Nothing under site-packages is edited: the installed
package stays byte-identical and a reader can diff it against PyPI. `install()` wraps
configure_optimizers on the task class so the first optimizer's first parameter group gains the
missing parameters. It is a no-op when the projection does not exist (categorical CPA builds no
pert_transformation) or when a future release already includes it, and it says on stdout exactly
what it added, so a run log records whether it was in effect.
"""

from __future__ import annotations


def install() -> str:
    """Patch CPA's task class in-process. Returns a one-line description for the run log."""
    from cpa._task import CPATrainingPlan

    if getattr(CPATrainingPlan, "_ivcbench_optimizer_shim", False):
        return "cpa optimizer shim: already installed"
    original = CPATrainingPlan.configure_optimizers

    def configure_optimizers(self):
        out = original(self)
        net = getattr(getattr(self, "module", None), "pert_network", None)
        proj = getattr(net, "pert_transformation", None)
        if proj is None:
            return out                      # categorical CPA: there is no projection to train
        optimizers = out[0] if isinstance(out, tuple) else out
        if not optimizers:
            return out
        group = optimizers[0].param_groups[0]
        known = {id(p) for o in optimizers for g in o.param_groups for p in g["params"]}
        missing = [p for p in proj.parameters() if p.requires_grad and id(p) not in known]
        if missing:
            group["params"].extend(missing)
            n = sum(p.numel() for p in missing)
            print(
                f"[cpa-shim] added pert_transformation to the autoencoder optimizer "
                f"({len(missing)} tensors, {n} parameters) -- upstream cpa-tools leaves the "
                f"chemical projection untrained",
                flush=True,
            )
        else:
            print("[cpa-shim] pert_transformation was already in an optimizer; nothing added", flush=True)
        return out

    CPATrainingPlan.configure_optimizers = configure_optimizers
    CPATrainingPlan._ivcbench_optimizer_shim = True
    return "cpa optimizer shim: installed on CPATrainingPlan.configure_optimizers"


def verify() -> dict:
    """Build a throwaway PerturbationNetwork and report optimizer membership. Evidence, not faith."""
    import torch.nn as nn
    from cpa._utils import PerturbationNetwork

    net = PerturbationNetwork(
        n_perts=7, n_latent=8, doser_type="logsigm", drug_embeddings=nn.Embedding(7, 16)
    )
    proj = {id(p) for p in net.pert_transformation.parameters()}
    covered = set(map(id, net.pert_embedding.parameters())) | set(
        map(id, net.dosers.parameters())
    )
    return {
        "projection_params": len(proj),
        "covered_by_upstream_optimizers": bool(proj & covered),
        "requires_grad": [p.requires_grad for p in net.pert_transformation.parameters()],
    }


if __name__ == "__main__":
    print(verify())
