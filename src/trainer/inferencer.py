import torch
from tqdm.auto import tqdm

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Inferencer(BaseTrainer):
    """
    Inference-only runner similar to a Trainer but without optimization.

    This class is used to evaluate a model on given datasets, compute metrics,
    and optionally save predictions. It omits training-specific logic such as
    optimizers, schedulers, or gradient steps.
    """

    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        text_encoder,
        save_path,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
    ):
        """
        Initialize the Inferencer.

        Args:
            model (nn.Module): PyTorch model used for inference.
            config (DictConfig): Experiment configuration.
            device (str): Target device for tensors and model.
            dataloaders (dict[str, DataLoader]): DataLoaders for evaluation datasets.
            text_encoder (CTCTextEncoder): Text encoder for ASR decoding.
            save_path (Path): Directory to save model predictions.
            metrics (dict[str, list[BaseMetric]] | None): Metrics definitions for inference.
            batch_transforms (dict[str, nn.Module] | None): Optional transforms applied per batch.
            skip_model_load (bool): If False, requires a pretrained checkpoint path.
        """
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.inferencer
        self.device = device
        self.model = model
        self.batch_transforms = batch_transforms
        self.text_encoder = text_encoder
        self.evaluation_dataloaders = dict(dataloaders)
        self.save_path = save_path
        self.metrics = metrics

        # Initialize metric tracker if metrics are provided
        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=None,
            )
        else:
            self.evaluation_metrics = None

        # Load pretrained model weights if required
        if not skip_model_load:
            self._from_pretrained(config.inferencer.get("from_pretrained"))

    def run_inference(self):
        """
        Run inference on all dataset partitions.

        Returns:
            dict[str, dict]: A mapping from partition name to computed logs.
        """
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs

    def process_batch(self, batch_idx, batch, metrics, part):
        """
        Process a single batch during inference.

        This method moves the batch to the target device, applies optional
        transformations, performs a forward pass, updates metrics, and
        optionally saves predictions to disk.

        Args:
            batch_idx (int): Current batch index.
            batch (dict): Input batch from DataLoader.
            metrics (MetricTracker | None): Metric tracker instance.
            part (str): Dataset partition name (e.g., "test", "dev").

        Returns:
            dict: Updated batch containing model outputs.
        """
        # Move batch to device and apply optional transformations
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        # Forward pass through the model
        outputs = self.model(**batch)
        batch.update(outputs)

        # Update evaluation metrics
        if metrics is not None:
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(**batch))

        # Save predictions to disk (example logic)
        batch_size = batch["logits"].shape[0]
        current_id = batch_idx * batch_size

        for i in range(batch_size):
            logits = batch["logits"][i].clone()
            label = batch["labels"][i].clone()
            pred_label = logits.argmax(dim=-1)
            output_id = current_id + i

            output = {"pred_label": pred_label, "label": label}

            if self.save_path is not None:
                torch.save(output, self.save_path / part / f"output_{output_id}.pth")

        return batch

    def _inference_part(self, part, dataloader):
        """
        Perform inference on a single dataset partition.

        Args:
            part (str): Partition name (e.g., "validation", "test").
            dataloader (DataLoader): DataLoader for the given partition.

        Returns:
            dict: Aggregated evaluation metrics for the partition.
        """
        self.is_train = False
        self.model.eval()
        self.evaluation_metrics.reset()

        # Ensure output directory exists
        if self.save_path is not None:
            (self.save_path / part).mkdir(exist_ok=True, parents=True)

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    part=part,
                    metrics=self.evaluation_metrics,
                )

        return self.evaluation_metrics.result()
