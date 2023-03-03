import os
import numpy as np
import sys
import torch
import logging
import model_settings
from stylegangeneratormodel import StyleGANGeneratorModel



class StyleGANGenerator():

  def __init__(self, logger=None):
    

    self.truncation_psi = model_settings.STYLEGAN_TRUNCATION_PSI
    self.truncation_layers = model_settings.STYLEGAN_TRUNCATION_LAYERS
    self.randomize_noise = model_settings.STYLEGAN_RANDOMIZE_NOISE
    self.model_specific_vars = ['truncation.truncation']

    self.model_name = "stylegan_ffhq"
    for key, val in model_settings.MODEL_POOL[self.model_name].items():
      setattr(self, key, val)
    self.use_cuda = model_settings.USE_CUDA
    self.batch_size = model_settings.MAX_IMAGES_ON_DEVICE
    self.logger = logger or get_temp_logger(self.model_name + '_generator')
    self.model = None
    self.run_device = 'cuda' if self.use_cuda else 'cpu'
    self.cpu_device = 'cpu'

  
    self.check_attr('gan_type')
    self.check_attr('latent_space_dim')
    self.check_attr('resolution')
    self.min_val = getattr(self, 'min_val', -1.0)
    self.max_val = getattr(self, 'max_val', 1.0)
    self.output_channels = getattr(self, 'output_channels', 3)
    self.channel_order = getattr(self, 'channel_order', 'RGB').upper()
    assert self.channel_order in ['RGB', 'BGR']

    
    self.build()
    if os.path.isfile(getattr(self, 'model_path', '')):
      self.load()
    else:
      self.logger.warning(f'No pre-trained model will be loaded!')

    
    assert self.model
    self.model.eval().to(self.run_device)

    self.num_layers = (int(np.log2(self.resolution)) - 1) * 2
    assert self.gan_type == 'stylegan'

    

  def build(self):
    self.check_attr('w_space_dim')
    self.check_attr('fused_scale')
    self.model = StyleGANGeneratorModel(
        resolution=self.resolution,
        w_space_dim=self.w_space_dim,
        fused_scale=self.fused_scale,
        output_channels=self.output_channels,
        truncation_psi=self.truncation_psi,
        truncation_layers=self.truncation_layers,
        randomize_noise=self.randomize_noise)

  def load(self):
    self.logger.info(f'Loading pytorch model from `{self.model_path}`.')
    state_dict = torch.load(self.model_path)
    for var_name in self.model_specific_vars:
      state_dict[var_name] = self.model.state_dict()[var_name]
    self.model.load_state_dict(state_dict)
    self.logger.info(f'Successfully loaded!')
    self.lod = self.model.synthesis.lod.to(self.cpu_device).tolist()
    self.logger.info(f'  `lod` of the loaded model is {self.lod}.')

  def sample(self, num, latent_space_type='Z'):

    latent_space_type = latent_space_type.upper()
    if latent_space_type == 'Z':
      latent_codes = np.random.randn(num, self.latent_space_dim)
    elif latent_space_type == 'W':
      latent_codes = np.random.randn(num, self.w_space_dim)
    elif latent_space_type == 'WP':
      latent_codes = np.random.randn(num, self.num_layers, self.w_space_dim)
    else:
      raise ValueError(f'Latent space type `{latent_space_type}` is invalid!')

    return self.preprocess(latent_codes.astype(np.float32),
                           latent_space_type)

  def preprocess(self, latent_codes, latent_space_type='Z'):

    if not isinstance(latent_codes, np.ndarray):
      raise ValueError(f'Latent codes should be with type `numpy.ndarray`!')

    latent_space_type = latent_space_type.upper()
    if latent_space_type == 'Z':
      latent_codes = latent_codes.reshape(-1, self.latent_space_dim)
      norm = np.linalg.norm(latent_codes, axis=1, keepdims=True)
      latent_codes = latent_codes / norm * np.sqrt(self.latent_space_dim)
    elif latent_space_type == 'W':
      latent_codes = latent_codes.reshape(-1, self.w_space_dim)
    elif latent_space_type == 'WP':
      latent_codes = latent_codes.reshape(-1, self.num_layers, self.w_space_dim)
    else:
      raise ValueError(f'Latent space type `{latent_space_type}` is invalid!')

    return latent_codes.astype(np.float32)


  def synthesize(self,
                 latent_codes,
                 latent_space_type='Z',
                 generate_style=False,
                 generate_image=True):
    if not isinstance(latent_codes, np.ndarray):
      raise ValueError(f'Latent codes should be with type `numpy.ndarray`!')

    results = {}

    latent_space_type = latent_space_type.upper()
    latent_codes_shape = latent_codes.shape
   
    if latent_space_type == 'Z':
      if not (len(latent_codes_shape) == 2 and
              latent_codes_shape[0] <= self.batch_size and
              latent_codes_shape[1] == self.latent_space_dim):
        raise ValueError(f'Latent_codes should be with shape [batch_size, '
                         f'latent_space_dim], where `batch_size` no larger '
                         f'than {self.batch_size}, and `latent_space_dim` '
                         f'equal to {self.latent_space_dim}!\n'
                         f'But {latent_codes_shape} received!')
      zs = torch.from_numpy(latent_codes).type(torch.FloatTensor)
      zs = zs.to(self.run_device)
      ws = self.model.mapping(zs)
      wps = self.model.truncation(ws)
      results['z'] = latent_codes
      results['w'] = self.get_value(ws)
      results['wp'] = self.get_value(wps)
   
    elif latent_space_type == 'W':
      if not (len(latent_codes_shape) == 2 and
              latent_codes_shape[0] <= self.batch_size and
              latent_codes_shape[1] == self.w_space_dim):
        raise ValueError(f'Latent_codes should be with shape [batch_size, '
                         f'w_space_dim], where `batch_size` no larger than '
                         f'{self.batch_size}, and `w_space_dim` equal to '
                         f'{self.w_space_dim}!\n'
                         f'But {latent_codes_shape} received!')
      ws = torch.from_numpy(latent_codes).type(torch.FloatTensor)
      ws = ws.to(self.run_device)
      wps = self.model.truncation(ws)
      results['w'] = latent_codes
      results['wp'] = self.get_value(wps)
    
    elif latent_space_type == 'WP':
      if not (len(latent_codes_shape) == 3 and
              latent_codes_shape[0] <= self.batch_size and
              latent_codes_shape[1] == self.num_layers and
              latent_codes_shape[2] == self.w_space_dim):
        raise ValueError(f'Latent_codes should be with shape [batch_size, '
                         f'num_layers, w_space_dim], where `batch_size` no '
                         f'larger than {self.batch_size}, `num_layers` equal '
                         f'to {self.num_layers}, and `w_space_dim` equal to '
                         f'{self.w_space_dim}!\n'
                         f'But {latent_codes_shape} received!')
      wps = torch.from_numpy(latent_codes).type(torch.FloatTensor)
      wps = wps.to(self.run_device)
      results['wp'] = latent_codes
    else:
      raise ValueError(f'Latent space type `{latent_space_type}` is invalid!')

    if generate_style:
      for i in range(self.num_layers):
        style = self.model.synthesis.__getattr__(
            f'layer{i}').epilogue.style_mod.dense(wps[:, i, :])
        results[f'style{i:02d}'] = self.get_value(style)

    if generate_image:
      images = self.model.synthesis(wps)
      results['image'] = self.get_value(images)
    
    if 'image' in results:
      results['image'] = self.postprocess(results['image'])
    
    return results
  
  def check_attr(self, attr_name):
    if not hasattr(self, attr_name):
      raise AttributeError(
          f'`{attr_name}` is missing for model `{self.model_name}`!')

  def get_value(self, tensor):
    print(tensor.shape)
    if isinstance(tensor, np.ndarray):
      return tensor
    if isinstance(tensor, torch.Tensor):
      return tensor.to(self.cpu_device).detach().numpy()
    raise ValueError(f'Unsupported input type `{type(tensor)}`!')

  
  def postprocess(self, images):
    if not isinstance(images, np.ndarray):
      raise ValueError(f'Images should be with type `numpy.ndarray`!')

    images_shape = images.shape
    if len(images_shape) != 4 or images_shape[1] not in [1, 3]:
      raise ValueError(f'Input should be with shape [batch_size, channel, '
                       f'height, width], where channel equals to 1 or 3. '
                       f'But {images_shape} is received!')
    images = (images - self.min_val) * 255 / (self.max_val - self.min_val)
    images = np.clip(images + 0.5, 0, 255).astype(np.uint8)
    images = images.transpose(0, 2, 3, 1)
    if self.channel_order == 'BGR':
      images = images[:, :, :, ::-1]

    return images    
  
  def get_batch_inputs(self, latent_codes):
    total_num = latent_codes.shape[0]
    for i in range(0, total_num, self.batch_size):
      yield latent_codes[i:i + self.batch_size]

def get_temp_logger(logger_name='logger'):
  if not logger_name:
    raise ValueError(f'Input `logger_name` should not be empty!')

  logger = logging.getLogger(logger_name)
  if not logger.hasHandlers():
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

  return logger
