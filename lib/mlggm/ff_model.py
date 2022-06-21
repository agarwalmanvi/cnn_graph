import pytorch_lightning as pl
import torch


class FF(pl.LightningModule):
    def __init__(
            self,
            input_size: int = 6000,
            hidden_size: int = 1024,
            output_size: int = 1,
            bias: bool = True,
            lr: float = 1e-3,
            **kwargs
    ):
        super(FF, self).__init__()

        self.save_hyperparameters()

        self.loss_fn = torch.nn.BCEWithLogitsLoss()

        self.hidden_layer = torch.nn.Linear(
            in_features=self.hparams.input_size,
            out_features=self.hparams.hidden_size,
            bias=self.hparams.bias
        )

        self.nonlin = torch.nn.Sigmoid()

        self.readout_layer = torch.nn.Linear(
            in_features=self.hparams.hidden_size,
            out_features=self.hparams.output_size,
            bias=self.hparams.bias
        )

    def forward(self, x):
        # batch x input_size -> batch x hidden_size
        out = self.hidden_layer(x)
        out = self.nonlin(out)
        #                   -> batch x output_size
        out = self.readout_layer(out)

        return out

    def training_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.RMSprop(self.parameters(), lr=self.hparams.lr)
        return opt