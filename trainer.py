import datetime
import json
from os import path, makedirs

import numpy as np
import torch
from tqdm import tqdm

BASE_DIR = path.dirname(path.abspath(__file__))


class EpochSeq2SeqTrainer:
    def __init__(self, model, train_dataloader, val_dataloader, loss_function,
                 metric_function, optimizer, logger, run_name, save_config,
                 save_checkpoint, config):
        self.config = config
        self.device = torch.device(self.config["device"])
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.loss_function = loss_function.to(self.device)
        self.metric_function = metric_function
        self.optimizer = optimizer
        # Q: What is a clip grad?
        # A: It clips graidents when they exceed a set value
        self.clip_grads = self.config["clip_grads"]

        self.logger = logger
        self.checkpoint_dir = path.join(BASE_DIR, "checkpoints", run_name)

        if not path.exists(self.checkpoint_dir):
            makedirs(self.checkpoint_dir)

        if save_config is None:
            config_filepath = path.join(self.checkpoint_dir, "config.json")
        else:
            config_filepath = save_config

        # I: I think this idea of saving config files is really nice.
        with open(config_filepath, "w") as config_file:
            json.dump(self.config, config_file)

        self.print_every = self.config["print_every"]
        self.save_every = self.config["save_every"]

        self.epoch = 0
        self.history = []

        self.start_time = datetime.datetime.now()

        self.best_val_metric = None
        self.best_checkpoint_filepath = None

        self.save_checkpoint = save_checkpoint
        self.save_format = "epoch={epoch:0>3}-val_loss={val_loss:<.3}-val_metrics={val_metrics}.pth"
        self.log_format = (
            "Epoch: {epoch:>3} "
            "Progress: {progress:<.1%} "
            "Elapsed: {elapsed} "
            "Examples/second: {per_second:<.1} "
            "Train Loss: {train_loss:<.6} "
            "Val Loss: {val_loss:<.6} "
            "Train Metrics: {train_metrics} "
            "Val Metrics: {val_metrics} "
            "Learning rate: {current_lr:<.4} "
        )

    def run(self, epochs=10):

        for epoch in range(self.epoch, epochs + 1):
            self.epoch = epoch
            self.model.train()

            epoch_start_time = datetime.datetime.now()
            train_epoch_loss, train_epoch_metrics = self.run_epoch(
                self.train_dataloader, mode="train")
            epoch_end_time = datetime.datetime.now()

            self.model.eval()
            val_epoch_loss, val_epoch_metrics = self.run_epoch(self.val_dataloader, mode="val")

            if epoch % self.print_every == 0 and self.logger:
                per_second = len(self.train_dataloader.dataset) / \
                             ((epoch_end_time - epoch_start_time).seconds + 1)
                # I: I did not know one could access the current learning rate this way.
                current_lr = self.optimizer.param_groups[0]["lr"]
                log_message = self.log_format.format(
                    epoch=epoch, progress=epoch/epochs,
                    per_second=per_second, train_loss=train_epoch_loss,
                    val_loss=val_epoch_loss,
                    train_metrics=[round(metric, 4) for metric in train_epoch_metrics],
                    val_metrics=[round(metric, 4) for metric in val_epoch_metrics],
                    current_lr=current_lr,
                    elapsed=self._elapsed_time()
                )
                self.logger.info(log_message)

            if epoch % self.save_every == 0:
                self._save_model(epoch, train_epoch_loss, val_epoch_loss,
                                 train_epoch_metrics, val_epoch_metrics)

    def run_epoch(self, dataloader, mode="train"):
        batch_losses = []
        batch_counts = []
        batch_metrics = []

        for sources, inputs, targets in tqdm(dataloader):
            sources, inputs, targets = sources.to(self.device), inputs.to(self.device), \
                                       targets.to(self.device)
            # Q: I need to understand properly how the source, inputs and targets are used in training and in
            # evaluation. This is because during inference, inputs and targets won't be available.
            # A: The source text is passed into the encoder which creates the keys and values.
            # The decoder takes in a first the special token <StartSent>, and uses this in combination
            # with the keys and values from the encoder to predict the first translated word.
            # Then it takes in that <StartSent> token and the first word, then uses this to predict the
            # second translated word and so on and so forth.
            outputs = self.model(sources, inputs)

            batch_loss, batch_count = self.loss_function(outputs, targets)

            # I: This is a really nice way to only do backprop for train purposes.
            if mode == "train":
                self.optimizer.zero_grad()
                # I: First time I am seeing clip grad being done. I think it's really cool
                # https://stackoverflow.com/questions/54716377/how-to-do-gradient-clipping-in-pytorch
                batch_loss.backward()
                if self.clip_grads:
                    torch.nn.utils.clip_grad_norm(self.model.parameters(), 1)
                self.optimizer.step()

            batch_losses.append(batch_loss.item())
            batch_counts.append(batch_count)

            batch_metric, batch_metric_count = self.metric_function(outputs, targets)
            batch_metrics.append(batch_metric)

            assert batch_count == batch_metric_count

            if self.epoch == 0:
                return float("inf"), [float("inf")]

        epoch_loss = sum(batch_losses) / sum(batch_counts)
        epoch_accuracy = sum(batch_metrics) / sum(batch_counts)
        # Q: What is perplexity?
        epoch_perplexity = float(np.exp(epoch_loss))
        epoch_metrics = [epoch_perplexity, epoch_accuracy]

        return epoch_loss, epoch_metrics

    def _elapsed_time(self):
        now = datetime.datetime.now()
        elapsed = now - self.start_time
        return str(elapsed).split(".")[0]

    def _save_model(self, epoch, train_epoch_loss, val_epoch_loss,
                    train_epoch_metrics, val_epoch_metrics):
        checkpoint_filename = self.save_format.format(
            epoch=epoch,
            val_loss=val_epoch_loss,
            val_metrics="-".join(['{:<.3}'.format(v) for v in val_epoch_metrics])
        )

        if self.save_checkpoint is None:
            checkpoint_filepath = path.join(self.checkpoint_dir, checkpoint_filename)
        else:
            checkpoint_filepath = self.save_checkpoint

        save_state = {
            "epoch": epoch,
            "train_loss": train_epoch_loss,
            "train_metrics": train_epoch_metrics,
            "val_loss": val_epoch_loss,
            "val_metrics": val_epoch_metrics,
            "checkpoint": checkpoint_filepath
        }

        if self.epoch > 0:
            torch.save(self.model.state_dict(), checkpoint_filepath)
            self.history.append(save_state)

        representative_val_metric = val_epoch_metrics[0]
        # Q: Should this be self.best_val_metric > representative_val_metric or the other way around?
        if self.best_val_metric is None or self.best_val_metric > representative_val_metric:
            self.best_val_metric = representative_val_metric
            self.val_loss_at_best = val_epoch_loss
            self.train_loss_at_best = train_epoch_loss
            self.train_metrics_at_best = train_epoch_metrics
            self.val_metrics_at_best = val_epoch_metrics
            self.best_checkpoint_filepath = checkpoint_filepath

        if self.logger:
            self.logger.info("Saved model to {}".format(checkpoint_filepath))
            self.logger.info("Current best model is {}".format(self.best_checkpoint_filepath))
