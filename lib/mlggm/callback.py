from pytorch_lightning.callbacks import Callback

class MetricTracker(Callback):

  def __init__(self):
    self.elogs_collection = []

  def on_validation_epoch_end(self, trainer, module):
    elogs = trainer.logged_metrics
    self.elogs_collection.append(elogs)