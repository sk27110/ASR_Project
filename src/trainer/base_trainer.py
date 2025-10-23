from abc import abstractmethod

import torch
from numpy import inf
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from src.datasets.data_utils import inf_loop
from src.metrics.tracker import MetricTracker
from src.utils.io_utils import ROOT_PATH


class BaseTrainer:
    """
    Abstract base class that defines the common training logic for PyTorch models.

    This class manages the full training loop, evaluation, checkpointing,
    learning rate scheduling, metric tracking, and early stopping.
    It is meant to be subclassed for specific training workflows that
    define their own `process_batch` and `_log_batch` methods.
    """

    def __init__(
        self,
        model,
        criterion,
        metrics,
        optimizer,
        lr_scheduler,
        text_encoder,
        config,
        device,
        dataloaders,
        logger,
        writer,
        epoch_len=None,
        skip_oom=True,
        batch_transforms=None,
    ):
        """
        Initialize the base trainer.

        Args:
            model (nn.Module): PyTorch model to be trained.
            criterion (nn.Module): Loss function used for training.
            metrics (dict): Dictionary of metric trackers for 'train' and 'inference' modes.
                Each metric is an instance of `src.metrics.BaseMetric`.
            optimizer (torch.optim.Optimizer): Optimizer for model parameters.
            lr_scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
            text_encoder (CTCTextEncoder): Text encoder instance for preprocessing/decoding text.
            config (DictConfig): Experiment configuration containing training parameters.
            device (str or torch.device): Device where the model and tensors are allocated.
            dataloaders (dict[str, torch.utils.data.DataLoader]): DataLoaders for 'train',
                'val', and possibly other datasets.
            logger (logging.Logger): Logger instance for console/file logging.
            writer (WandBWriter | CometMLWriter): Experiment tracking writer for logging metrics.
            epoch_len (int | None): Number of steps per epoch. If None, full epoch (len(dataloader)) is used.
            skip_oom (bool): Whether to skip batches that cause OutOfMemoryError.
            batch_transforms (dict[str, Callable] | None): Optional transforms applied to full batches,
                specified separately for 'train' and 'inference' modes.
        """
        self.is_train = True
        self.config = config
        self.cfg_trainer = self.config.trainer
        self.device = device
        self.skip_oom = skip_oom

        self.logger = logger
        self.log_step = config.trainer.get("log_step", 50)

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.text_encoder = text_encoder
        self.batch_transforms = batch_transforms

        # Define dataloaders
        self.train_dataloader = dataloaders["train"]
        if epoch_len is None:
            # Epoch-based training
            self.epoch_len = len(self.train_dataloader)
        else:
            # Iteration-based training (infinite loop)
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.epoch_len = epoch_len

        # Define evaluation dataloaders
        self.evaluation_dataloaders = {
            k: v for k, v in dataloaders.items() if k != "train"
        }

        # Epoch configuration
        self._last_epoch = 0
        self.start_epoch = 1
        self.epochs = self.cfg_trainer.n_epochs

        # Monitoring and checkpointing setup
        self.save_period = self.cfg_trainer.save_period
        self.monitor = self.cfg_trainer.get("monitor", "off")

        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]
            self.mnt_best = inf if self.mnt_mode == "min" else -inf
            self.early_stop = max(self.cfg_trainer.get("early_stop", inf), 1)

        # Visualization and logging
        self.writer = writer
        self.metrics = metrics

        self.train_metrics = MetricTracker(
            *self.config.writer.loss_names,
            "grad_norm",
            *[m.name for m in self.metrics["train"]],
            writer=self.writer,
        )
        self.evaluation_metrics = MetricTracker(
            *self.config.writer.loss_names,
            *[m.name for m in self.metrics["inference"]],
            writer=self.writer,
        )

        # Checkpoint directory
        self.checkpoint_dir = (
            ROOT_PATH / config.trainer.save_dir / config.writer.run_name
        )

        # Resume or load pretrained weights if specified
        if config.trainer.get("resume_from") is not None:
            self._resume_checkpoint(self.checkpoint_dir / config.trainer.resume_from)
        if config.trainer.get("from_pretrained") is not None:
            self._from_pretrained(config.trainer.get("from_pretrained"))

    def train(self):
        """
        Entry point for model training.
        Handles graceful interruption (Ctrl+C) by saving a checkpoint.
        """
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self.logger.info("Saving model on keyboard interrupt.")
            self._save_checkpoint(self._last_epoch, save_best=False)
            raise e

    def _train_process(self):
        """
        Core training process over multiple epochs:
        - Trains model for each epoch.
        - Evaluates after each epoch.
        - Monitors performance and saves checkpoints.
        - Applies early stopping if needed.
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._last_epoch = epoch
            result = self._train_epoch(epoch)

            logs = {"epoch": epoch}
            logs.update(result)

            for key, value in logs.items():
                self.logger.info(f"    {key: 15s}: {value}")

            best, stop_process, not_improved_count = self._monitor_performance(
                logs, not_improved_count
            )

            if epoch % self.save_period == 0 or best:
                self._save_checkpoint(epoch, save_best=best, only_best=True)

            if stop_process:
                break

    def _train_epoch(self, epoch):
        """
        Performs training for a single epoch.

        Args:
            epoch (int): Current epoch number.

        Returns:
            dict: Aggregated logs containing average loss and metrics for the epoch.
        """
        self.is_train = True
        self.model.train()
        self.train_metrics.reset()
        self.writer.set_step((epoch - 1) * self.epoch_len)
        self.writer.add_scalar("epoch", epoch)

        for batch_idx, batch in enumerate(
            tqdm(self.train_dataloader, desc="train", total=self.epoch_len)
        ):
            try:
                batch = self.process_batch(batch, metrics=self.train_metrics)
            except torch.cuda.OutOfMemoryError:
                if self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    torch.cuda.empty_cache()
                    continue
                raise

            self.train_metrics.update("grad_norm", self._get_grad_norm())

            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.epoch_len + batch_idx)
                self.logger.debug(
                    f"Train Epoch: {epoch} {self._progress(batch_idx)} Loss: {batch['loss'].item(): .6f}"
                )
                self.writer.add_scalar(
                    "learning rate", self.lr_scheduler.get_last_lr()[0]
                )
                self._log_scalars(self.train_metrics)
                self._log_batch(batch_idx, batch)
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()

            if batch_idx + 1 >= self.epoch_len:
                break

        logs = last_train_metrics

        # Evaluate on all validation/test datasets
        for part, dataloader in self.evaluation_dataloaders.items():
            val_logs = self._evaluation_epoch(epoch, part, dataloader)
            logs.update({f"{part}_{name}": value for name, value in val_logs.items()})

        return logs

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Runs model evaluation on a given partition.

        Args:
            epoch (int): Current epoch number.
            part (str): Dataset partition name ('val', 'test', etc.).
            dataloader (DataLoader): DataLoader for the partition.

        Returns:
            dict: Aggregated evaluation metrics.
        """
        self.is_train = False
        self.model.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader), desc=part, total=len(dataloader)
            ):
                batch = self.process_batch(batch, metrics=self.evaluation_metrics)
            self.writer.set_step(epoch * self.epoch_len, part)
            self._log_scalars(self.evaluation_metrics)
            self._log_batch(batch_idx, batch, part)
        return self.evaluation_metrics.result()

    def _monitor_performance(self, logs, not_improved_count):
        """
        Monitors training performance and determines whether to save or stop.

        Args:
            logs (dict): Logged metrics for the current epoch.
            not_improved_count (int): Number of epochs without improvement.

        Returns:
            tuple[bool, bool, int]:
                - best: whether the metric improved,
                - stop_process: whether to trigger early stopping,
                - not_improved_count: updated counter.
        """
        best = False
        stop_process = False
        if self.mnt_mode != "off":
            try:
                improved = (
                    logs[self.mnt_metric] <= self.mnt_best
                    if self.mnt_mode == "min"
                    else logs[self.mnt_metric] >= self.mnt_best
                )
            except KeyError:
                self.logger.warning(
                    f"Metric '{self.mnt_metric}' not found. Monitoring disabled."
                )
                self.mnt_mode = "off"
                improved = False

            if improved:
                self.mnt_best = logs[self.mnt_metric]
                not_improved_count = 0
                best = True
            else:
                not_improved_count += 1

            if not_improved_count >= self.early_stop:
                self.logger.info(
                    f"Validation performance did not improve for {self.early_stop} epochs. Stopping."
                )
                stop_process = True

        return best, stop_process, not_improved_count

    def move_batch_to_device(self, batch, non_blocking=False):
        """
        Moves specified tensors in a batch to the training device.

        Args:
            batch (dict): Batch containing tensors.
            non_blocking (bool): Use non-blocking memory transfer if possible.

        Returns:
            dict: Batch with tensors moved to device.
        """
        for tensor_for_device in self.cfg_trainer.device_tensors:
            batch[tensor_for_device] = batch[tensor_for_device].to(
                self.device, non_blocking=non_blocking
            )
        return batch

    def transform_batch(self, batch):
        """
        Apply batch-level transforms (e.g., augmentations) to the entire batch.

        Args:
            batch (dict): Original batch.

        Returns:
            dict: Transformed batch.
        """
        transform_type = "train" if self.is_train else "inference"
        transforms = self.batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                batch[transform_name] = transforms[transform_name](
                    batch[transform_name]
                )
        return batch

    def _clip_grad_norm(self):
        """
        Clip gradient norms to avoid exploding gradients.
        Uses the value defined in config.trainer.max_grad_norm.
        """
        if self.config["trainer"].get("max_grad_norm", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["max_grad_norm"]
            )

    @torch.no_grad()
    def _get_grad_norm(self, norm_type=2):
        """
        Compute total gradient norm for all parameters.

        Args:
            norm_type (float | str): Order of the norm.

        Returns:
            float: Calculated gradient norm.
        """
        parameters = [p for p in self.model.parameters() if p.grad is not None]
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
            norm_type,
        )
        return total_norm.item()

    def _progress(self, batch_idx):
        """
        Format current training progress as string.

        Args:
            batch_idx (int): Current batch index.

        Returns:
            str: Progress string with percentage.
        """
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.epoch_len
        return base.format(current, total, 100.0 * current / total)

    @abstractmethod
    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from a batch (to be implemented in subclasses).

        Args:
            batch_idx (int): Batch index.
            batch (dict): Processed batch.
            mode (str): 'train' or 'inference'.
        """
        raise NotImplementedError()

    def _log_scalars(self, metric_tracker: MetricTracker):
        """
        Log scalar metrics from a MetricTracker instance.

        Args:
            metric_tracker (MetricTracker): Metric tracker with current values.
        """
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))

    def _save_checkpoint(self, epoch, save_best=False, only_best=False):
        """
        Save model, optimizer, and scheduler state to a checkpoint file.

        Args:
            epoch (int): Current epoch number.
            save_best (bool): Whether to also save 'model_best.pth'.
            only_best (bool): If True, only save when best.
        """
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }
        filename = str(self.checkpoint_dir / f"checkpoint-epoch{epoch}.pth")
        if not (only_best and save_best):
            torch.save(state, filename)
            if self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(filename, str(self.checkpoint_dir.parent))
            self.logger.info(f"Saving checkpoint: {filename} ...")
        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            if self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(best_path, str(self.checkpoint_dir.parent))
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume model and optimizer state from a saved checkpoint.

        Args:
            resume_path (str): Path to checkpoint file.
        """
        resume_path = str(resume_path)
        self.logger.info(f"Loading checkpoint: {resume_path} ...")
        checkpoint = torch.load(resume_path, self.device)
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        if checkpoint["config"]["model"] != self.config["model"]:
            self.logger.warning(
                "Architecture configuration differs from checkpoint; loading may fail."
            )
        self.model.load_state_dict(checkpoint["state_dict"])

        if (
            checkpoint["config"]["optimizer"] != self.config["optimizer"]
            or checkpoint["config"]["lr_scheduler"] != self.config["lr_scheduler"]
        ):
            self.logger.warning(
                "Optimizer or scheduler differs; not resuming their states."
            )
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        self.logger.info(f"Checkpoint loaded. Resuming from epoch {self.start_epoch}.")

    def _from_pretrained(self, pretrained_path):
        """
        Load model weights from a pretrained checkpoint (model only).

        Args:
            pretrained_path (str): Path to pretrained model checkpoint.
        """
        pretrained_path = str(pretrained_path)
        if hasattr(self, "logger"):
            self.logger.info(f"Loading model weights from: {pretrained_path} ...")
        else:
            print(f"Loading model weights from: {pretrained_path} ...")

        checkpoint = torch.load(pretrained_path, self.device)
        if checkpoint.get("state_dict") is not None:
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
