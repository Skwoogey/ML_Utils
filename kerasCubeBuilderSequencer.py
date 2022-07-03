import tensorflow as tf
import HSiCubeBuilder as cb

class HSICubePredictSequencer(tf.keras.utils.Sequence):
	def __init__(self, HSIIH, augment_variant=False, augment_transform=False, batch_size = None):
		self.HSIIH = HSIIH
		self.augment_variant = augment_variant
		self.augment_transform = augment_transform
		self.samples_per_pixel = 1
		if self.augment_transform:
			self.samples_per_pixel *= 8
		if self.augment_variant:
			self.samples_per_pixel *= len(self.HSIIH.variant_keys)
		self.batch_size = batch_size
		if self.batch_size == None:
			self.batch_size = self.samples_per_pixel

		self.samples = []
		for pxl in self.HSIIH.labeled_pixels:
			self.HSIIH.populatePixelWithVariants(pxl, self.augment_transform)
			self.samples += pxl.get(self.augment_variant, self.augment_transform)


	def __len__(self):
		return int(math.ceil(len(self.samples) /  self.batch_size))

	def __getitem__(self, idx):
		batch_x = self.samples[idx * self.batch_size : (idx + 1) * self.batch_size]

		batch_x = np.stack(batch_x, axis = 0)
		
		return batch_x
