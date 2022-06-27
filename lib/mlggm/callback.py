from pytorch_lightning.callbacks import Callback

class MetricTracker(Callback):

  def __init__(self):
    self.collection = []
    self.elogs_collection = []

  # def on_validation_batch_end(self, trainer, module, outputs):
  #   vacc = outputs[-1] # you can access them here
  #   self.collection.append(vacc) # track them

  def on_validation_epoch_end(self, trainer, module):
    elogs = trainer.logged_metrics # access it here
    self.elogs_collection.append(elogs)
    # do whatever is needed
