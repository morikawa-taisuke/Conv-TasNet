import pytorch_lightning as pl

class ConvTasNetLightning(pl.LightningModule):
	def __init__(self, model, optimizer, loss_function, config):
		super().__init__()
		self.model = model
		self.optimizer = optimizer
		self.loss_function = loss_function
		self.config = config
		self.loss_func_name = config["train"]["loss_function"]

	def forward(self, x):
		return self.model(x)

	def _calc_loss(self, mix_data, target_data):
		estimate_data = self(mix_data)
		
		if self.loss_func_name in ["SISDR", "SISNR"] and mix_data.size(0) > 1:
			loss = 0
			for i in range(mix_data.size(0)):
				loss += self.loss_function(estimate_data[i].unsqueeze(0), target_data[i].unsqueeze(0))
			loss /= mix_data.size(0)
		else:
			loss = self.loss_function(estimate_data, target_data)
			
		return loss

	def training_step(self, batch, batch_idx):
		mix_data, target_data = batch
		loss = self._calc_loss(mix_data, target_data)
		self.log("Loss/train", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
		return loss

	def validation_step(self, batch, batch_idx):
		mix_data, target_data = batch
		loss = self._calc_loss(mix_data, target_data)
		self.log("Loss/validation", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
		return loss

	def configure_optimizers(self):
		return self.optimizer
