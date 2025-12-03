import io

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from sklearn.decomposition import PCA

from .utils import send_batch_to_device

plt.switch_backend("Agg")


def has_image_loggers(loggers):
    """Checks whether any image loggers are available.

    Parameters
    ----------
    loggers : obj or list[obj]
        An object or list of loggers to search.
    """
    logger_list = loggers if isinstance(loggers, list) else [loggers]
    for logger in logger_list:
        if isinstance(logger, pl.loggers.TensorBoardLogger):
            return True
        elif isinstance(logger, pl.loggers.WandbLogger):
            return True
    return False


def log_figure(loggers, name, fig, step):
    """Logs a figure image to all available image loggers.

    Parameters
    ----------
    loggers : obj or list[obj]
        An object or list of loggers
    name : str
        The name to use for the logged figure
    fig : matplotlib.figure.Figure
        The figure to log
    step : int
        The step to associate with the logged figure
    """
    # Save figure image to in-memory buffer
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format="png")
    image = Image.open(img_buf)
    # Distribute image to all image loggers
    logger_list = loggers if isinstance(loggers, list) else [loggers]
    for logger in logger_list:
        if isinstance(logger, pl.loggers.TensorBoardLogger):
            logger.experiment.add_figure(name, fig, step)
        elif isinstance(logger, pl.loggers.WandbLogger):
            logger.log_image(name, [image], step)
    img_buf.close()


class RasterPlot(pl.Callback):
    """Plots validation spiking data side-by-side with
    inferred inputs and rates and logs to tensorboard.
    """

    def __init__(self, split="valid", n_samples=3, log_every_n_epochs=100):
        """Initializes the callback.
        Parameters
        ----------
        n_samples : int, optional
            The number of samples to plot, by default 3
        log_every_n_epochs : int, optional
            The frequency with which to plot and log, by default 100
        """
        assert split in ["train", "valid"]
        self.split = split
        self.n_samples = n_samples
        self.log_every_n_epochs = log_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        """Logs plots at the end of the validation epoch.
        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The trainer currently handling the model.
        pl_module : pytorch_lightning.LightningModule
            The model currently being trained.
        """
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        # Check for any image loggers
        if not has_image_loggers(trainer.loggers):
            return
        # Get data samples from the dataloaders
        if self.split == "valid":
            dataloader = trainer.datamodule.val_dataloader()
        else:
            dataloader = trainer.datamodule.train_dataloader(shuffle=False)
        batch = next(iter(dataloader))
        # Determine which sessions are in the batch
        sessions = sorted(batch.keys())
        # Move data to the right device
        batch = send_batch_to_device(batch, pl_module.device)
        # Compute model output
        output = pl_module.predict_step(
            batch=batch,
            batch_ix=None,
            sample_posteriors=True,
        )
        # Discard the extra data - only the SessionBatches are relevant here
        batch = {s: b[0] for s, b in batch.items()}
        # Log a few example outputs for each session
        for s in sessions:
            # Convert everything to numpy
            encod_data = batch[s].encod_data.detach().cpu().numpy()
            recon_data = batch[s].recon_data.detach().cpu().numpy()
            truth = batch[s].truth.detach().cpu().numpy()
            means = output[s].output_params.detach().cpu().numpy()
            inputs = output[s].gen_inputs.detach().cpu().numpy()
            # Compute data sizes
            _, steps_encod, neur_encod = encod_data.shape
            _, steps_recon, neur_recon = recon_data.shape
            # Decide on how to plot panels
            if np.all(np.isnan(truth)):
                plot_arrays = [recon_data, means, inputs]
                height_ratios = [3, 3, 1]
            else:
                plot_arrays = [recon_data, truth, means, inputs]
                height_ratios = [3, 3, 3, 1]
            # Create subplots
            fig, axes = plt.subplots(
                len(plot_arrays),
                self.n_samples,
                sharex=True,
                sharey="row",
                figsize=(3 * self.n_samples, 10),
                gridspec_kw={"height_ratios": height_ratios},
            )
            for i, ax_col in enumerate(axes.T):
                for j, (ax, array) in enumerate(zip(ax_col, plot_arrays)):
                    if j < len(plot_arrays) - 1:
                        ax.imshow(array[i].T, interpolation="none", aspect="auto")
                        ax.vlines(steps_encod, 0, neur_recon, color="orange")
                        ax.hlines(neur_encod, 0, steps_recon, color="orange")
                        ax.set_xlim(0, steps_recon)
                        ax.set_ylim(0, neur_recon)
                    else:
                        ax.plot(array[i])
            plt.tight_layout()
            # Log the figure
            log_figure(
                trainer.loggers,
                f"{self.split}/raster_plot/sess{s}",
                fig,
                trainer.global_step,
            )


class TrajectoryPlot(pl.Callback):
    """Plots the top-3 PC's of the latent trajectory for
    all samples in the validation set and logs to tensorboard.
    """

    def __init__(self, log_every_n_epochs=100):
        """Initializes the callback.

        Parameters
        ----------
        log_every_n_epochs : int, optional
            The frequency with which to plot and log, by default 100
        """
        self.log_every_n_epochs = log_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        """Logs plots at the end of the validation epoch.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The trainer currently handling the model.
        pl_module : pytorch_lightning.LightningModule
            The model currently being trained.
        """
        # Skip evaluation for most epochs to save time
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        # Check for any image loggers
        if not has_image_loggers(trainer.loggers):
            return
        # Get only the validation dataloaders
        pred_dls = trainer.datamodule.predict_dataloader()
        dataloaders = {s: dls["valid"] for s, dls in pred_dls.items()}
        # Compute outputs and plot for one session at a time
        for s, dataloader in dataloaders.items():
            latents = []
            for batch in dataloader:
                # Move data to the right device
                batch = send_batch_to_device({s: batch}, pl_module.device)
                # Perform the forward pass through the model
                output = pl_module.predict_step(batch, None, sample_posteriors=False)[s]
                latents.append(output.factors)
            latents = torch.cat(latents).detach().cpu().numpy()
            # Reduce dimensionality if necessary
            n_samp, n_step, n_lats = latents.shape
            if n_lats > 3:
                latents_flat = latents.reshape(-1, n_lats)
                pca = PCA(n_components=3)
                latents = pca.fit_transform(latents_flat)
                latents = latents.reshape(n_samp, n_step, 3)
                explained_variance = np.sum(pca.explained_variance_ratio_)
            else:
                explained_variance = 1.0
            # Create figure and plot trajectories
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection="3d")
            for traj in latents:
                ax.plot(*traj.T, alpha=0.2, linewidth=0.5)
            ax.scatter(*latents[:, 0, :].T, alpha=0.1, s=10, c="g")
            ax.scatter(*latents[:, -1, :].T, alpha=0.1, s=10, c="r")
            ax.set_title(f"explained variance: {explained_variance:.2f}")
            plt.tight_layout()
            # Log the figure
            log_figure(
                trainer.loggers,
                f"trajectory_plot/sess{s}",
                fig,
                trainer.global_step,
            )


class TestEval(pl.Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        test_batch = send_batch_to_device(
            trainer.datamodule.test_data[0][0], pl_module.device
        )
        _, esl, edd = test_batch.encod_data.shape
        test_output = pl_module(test_batch, output_means=False)[0]
        test_recon = pl_module.recon[0].compute_loss(
            test_batch.encod_data,
            test_output.output_params[:, :esl, :edd],
        )
        pl_module.log("test/recon", test_recon)

class MinSingularValueHistogram(pl.Callback):
    """
    Logs a histogram of minimum singular values of the readout Jacobian
    ∂rates/∂z. Uses JVP for efficiency.
    - Runs every `log_every_n_epochs`
    - Uses one validation batch
    - No graph construction (create_graph=False)
    """
    def __init__(
        self,
        sample_size=256,          # points in latent space
        col_sample_size=0,        # 0 => use all dims (capped)
        max_cols_cap=64,
        neuron_sample_size=0,     # subsample rate dims
        log_every_n_epochs=10,
        eps=1e-12,
    ):
        super().__init__()
        self.sample_size = sample_size
        self.col_sample_size = col_sample_size
        self.max_cols_cap = max_cols_cap
        self.neuron_sample_size = neuron_sample_size
        self.log_every_n_epochs = log_every_n_epochs
        self.eps = eps
    # ---------------- Utility funcs copied from loss class ---------------- #
    @staticmethod
    def _as_B1D(z):
        return z.view(z.shape[0], 1, z.shape[1])
    @staticmethod
    def _pick_tensor(x):
        if torch.is_tensor(x):
            return x
        if isinstance(x, dict):
            for k in ("rate","rates","mean","means","log_rate","pre_rate","loc"):
                if k in x and torch.is_tensor(x[k]):
                    return x[k]
            for v in x.values():
                if torch.is_tensor(v):
                    return v
        if isinstance(x, (tuple,list)):
            for v in x:
                if torch.is_tensor(v):
                    return v
        raise RuntimeError("readout returned no tensor")
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
        # else use all (capped)
        K = min(D, self.max_cols_cap)
        if K < D:
            return torch.randperm(D, device=device)[:K]
        return torch.arange(D, device=device)
    # ---------------- Singular value extraction ---------------- #
    def _compute_min_svals_for_area(self, model, z):
        """
        Compute min singular value of J = d rates / d z for each sample.
        """
        S, D = z.shape
        device = z.device
        def F(inp):
            return self._rates_from_z(model, inp)
        z_req = z.clone().detach().requires_grad_(True)
        rates0 = F(z_req)
        # neuron subsampling
        if (
            self.neuron_sample_size > 0
            and rates0.shape[1] > self.neuron_sample_size
        ):
            idxn = torch.randperm(rates0.shape[1], device=device)[:self.neuron_sample_size]
        else:
            idxn = None
        # latent columns
        idx_cols = self._choose_columns(D, device)
        K = idx_cols.numel()
        J_cols = []
        for d in idx_cols.tolist():
            v = torch.zeros_like(z_req)
            v[:, d] = 1.0
            _, jvp = torch.autograd.functional.jvp(
                F, z_req, v, create_graph=False
            )
            if idxn is not None:
                jvp = jvp[:, idxn]
            J_cols.append(jvp)
        # Stack: [S, Nsub, K]
        J = torch.stack(J_cols, dim=-1)
        min_svals = []
        for sidx in range(S):
            Js = J[sidx]   # [Nsub, K]
            try:
                svals = torch.linalg.svdvals(Js)
            except RuntimeError:
                svals = torch.linalg.svdvals(Js.cpu()).to(device)
            min_svals.append(svals.min().item())
        return min_svals
    # ---------------- Lightning hook ---------------- #
    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.log_every_n_epochs != 0:
            return
        if getattr(trainer, "sanity_checking", False):
            return
        if not has_image_loggers(trainer.loggers):
            return
        pl_module.eval()
        device = pl_module.device
       
        # We will collect min svs across TRAIN+VAL batches,
        # exactly matching your working extraction code.
        all_min_svals = []
        pred_dls = trainer.datamodule.predict_dataloader()
        dataloaders = {s: dls["valid"] for s, dls in pred_dls.items()}
        # Loop over both loaders
        for s, dataloader in dataloaders.items():
            for batch in dataloader:
                # Your exact format:
                # Move data to the right device
                batch = send_batch_to_device({s: batch}, pl_module.device)
                # Perform the forward pass through the model
                with torch.no_grad():
                    output = pl_module.predict_step(batch, None, sample_posteriors=False)[s]
                # Now grab latents 
                Z = output.factors.detach()  # [B,T,D]
                B, T, D = Z.shape
                zflat = Z.reshape(B*T, D)
                # Subsample to speed up
                if zflat.shape[0] > self.sample_size:
                    idx = torch.randperm(zflat.shape[0], device=device)[:self.sample_size]
                    z = zflat[idx]
                else:
                    z = zflat
                # Compute SVD(J)
                min_svals = self._compute_min_svals_for_area(pl_module, z)
                all_min_svals.extend(min_svals)
        # ---------------------------------------------------------
        # Plot aggregated histogram
        # ---------------------------------------------------------
        fig, ax = plt.subplots(1, 1, figsize=(8,5))
        if len(all_min_svals) > 0:
            ax.hist(
                np.array(all_min_svals),
                bins=40,
                alpha=0.6,
            )
        ax.set_title(f"Min Singular Values at Epoch {epoch}")
        ax.set_xlabel("Min singular value")
        ax.set_ylabel("Count")
        ax.legend()
        fig.tight_layout()
        log_figure(
            trainer.loggers,
            name=f"min_svals_histogram",
            fig=fig,
            step=trainer.global_step,
        )
        plt.close(fig)
