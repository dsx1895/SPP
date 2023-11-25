import torch
import warnings
from typing import Optional

from .. import separate
from ..masknn import activations
from ..utils.torch_utils import pad_x_to_y, script_if_tracing, jitable_shape
from ..utils.hub_utils import cached_download, SR_HASHTABLE
from ..utils.deprecation_utils import is_overridden, mark_deprecated

@script_if_tracing
def _unsqueeze_to_3d(x):
    """Normalize shape of `x` to [batch, n_chan, time]."""
    if x.ndim == 1:
        return x.reshape(1, 1, -1)
    elif x.ndim == 2:
        return x.unsqueeze(1)
    else:
        return x

class BaseModel(torch.nn.Module):
    """Base class for serializable models.

    Defines saving/loading procedures, and separation interface to `separate`.
    Need to overwrite the `forward` and `get_model_args` methods.

    Models inheriting from `BaseModel` can be used by :mod:`asteroid.separate`
    and by the `asteroid-infer` CLI. For models whose `forward` doesn't go from
    waveform to waveform tensors, overwrite `forward_wav` to return
    waveform tensors.

    Args:
        sample_rate (float): Operating sample rate of the model.
        in_channels: Number of input channels in the signal.
            If None, no checks will be performed.
    """

    def __init__(self, sample_rate: float, in_channels: Optional[int] = 1):
        super().__init__()
        self.__sample_rate = sample_rate
        self.in_channels = in_channels

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def sample_rate(self):
        """Operating sample rate of the model (float)."""
        return self.__sample_rate

    @sample_rate.setter
    def sample_rate(self, new_sample_rate: float):
        warnings.warn(
            "Other sub-components of the model might have a `sample_rate` "
            "attribute, be sure to modify them for consistency.",
            UserWarning,
        )
        self.__sample_rate = new_sample_rate

    def separate(self, *args, **kwargs):
        """Convenience for :func:`~asteroid.separate.separate`."""
        return separate.separate(self, *args, **kwargs)

    def torch_separate(self, *args, **kwargs):
        """Convenience for :func:`~asteroid.separate.torch_separate`."""
        return separate.torch_separate(self, *args, **kwargs)

    def numpy_separate(self, *args, **kwargs):
        """Convenience for :func:`~asteroid.separate.numpy_separate`."""
        return separate.numpy_separate(self, *args, **kwargs)

    def file_separate(self, *args, **kwargs):
        """Convenience for :func:`~asteroid.separate.file_separate`."""
        return separate.file_separate(self, *args, **kwargs)

    def forward_wav(self, wav, *args, **kwargs):
        """Separation method for waveforms.

        In case the network's `forward` doesn't have waveforms as input/output,
        overwrite this method to separate from waveform to waveform.
        Should return a single torch.Tensor, the separated waveforms.

        Args:
            wav (torch.Tensor): waveform array/tensor.
                Shape: 1D, 2D or 3D tensor, time last.
        """
        return self(wav, *args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_conf_or_path, *args, **kwargs):
        """Instantiate separation model from a model config (file or dict).

        Args:
            pretrained_model_conf_or_path (Union[dict, str]): model conf as
                returned by `serialize`, or path to it. Need to contain
                `model_args` and `state_dict` keys.
            *args: Positional arguments to be passed to the model.
            **kwargs: Keyword arguments to be passed to the model.
                They overwrite the ones in the model package.

        Returns:
            nn.Module corresponding to the pretrained model conf/URL.

        Raises:
            ValueError if the input config file doesn't contain the keys
                `model_name`, `model_args` or `state_dict`.
        """
        from . import get  # Avoid circular imports

        if isinstance(pretrained_model_conf_or_path, str):
            cached_model = cached_download(pretrained_model_conf_or_path)
            conf = torch.load(cached_model, map_location="cpu")
        else:
            conf = pretrained_model_conf_or_path

        if "model_name" not in conf.keys():
            raise ValueError(
                "Expected config dictionary to have field "
                "model_name`. Found only: {}".format(conf.keys())
            )
        if "state_dict" not in conf.keys():
            raise ValueError(
                "Expected config dictionary to have field "
                "state_dict`. Found only: {}".format(conf.keys())
            )
        if "model_args" not in conf.keys():
            raise ValueError(
                "Expected config dictionary to have field "
                "model_args`. Found only: {}".format(conf.keys())
            )
        conf["model_args"].update(kwargs)  # kwargs overwrite config.
        if "sample_rate" not in conf["model_args"] and isinstance(
            pretrained_model_conf_or_path, str
        ):
            conf["model_args"]["sample_rate"] = SR_HASHTABLE.get(
                pretrained_model_conf_or_path, None
            )
        # Attempt to find the model and instantiate it.
        try:
            model_class = get(conf["model_name"])
        except ValueError:  # Couldn't get the model, maybe custom.
            model = cls(*args, **conf["model_args"])  # Child class.
        else:
            model = model_class(*args, **conf["model_args"])
        model.load_state_dict(conf["state_dict"])
        return model

    def serialize(self):
        """Serialize model and output dictionary.

        Returns:
            dict, serialized model with keys `model_args` and `state_dict`.
        """
        import pytorch_lightning as pl  # Not used in torch.hub

        from .. import __version__ as asteroid_version  # Avoid circular imports

        model_conf = dict(
            model_name=self.__class__.__name__,
            state_dict=self.get_state_dict(),
            model_args=self.get_model_args(),
        )
        # Additional infos
        infos = dict()
        infos["software_versions"] = dict(
            torch_version=torch.__version__,
            pytorch_lightning_version=pl.__version__,
            asteroid_version=asteroid_version,
        )
        model_conf["infos"] = infos
        return model_conf

    def get_state_dict(self):
        """In case the state dict needs to be modified before sharing the model."""
        return self.state_dict()

    def get_model_args(self):
        """Should return args to re-instantiate the class."""
        raise NotImplementedError


class BaseEncoderMaskerDecoder(BaseModel):
    """Base class for encoder-masker-decoder separation models.

    Args:
        encoder (Encoder): Encoder instance.
        masker (nn.Module): masker network.
        decoder (Decoder): Decoder instance.
        encoder_activation (Optional[str], optional): Activation to apply after encoder.
            See ``asteroid.masknn.activations`` for valid values.
    """
    #encoder_enhanced
    def __init__(self, 
                 enh_encoder,
                 enh_masker,
                 enh_decoder,
                 sep_encoder,
                 sep_masker,
                 sep_decoder,
                 ssl_model=None,
                 encoder_activation=None):
        super().__init__(sample_rate=getattr(enh_encoder, "sample_rate", None))
        self.enh_encoder = enh_encoder
        self.enh_masker = enh_masker
        self.enh_decoder = enh_decoder
        self.sep_encoder = sep_encoder
        self.sep_masker = sep_masker
        self.sep_decoder = sep_decoder
        self.ssl_model = ssl_model
        self.encoder_activation = encoder_activation
        self.enh_enc_activation = activations.get(encoder_activation or "linear")()
        self.sep_enc_activation = activations.get(encoder_activation or "linear")()
    
    def forward(self, wav):
    
        separated_wav = self.forward_sep(wav)
        enhanced_wav = self.forward_enh(separated_wav)
        return enhanced_wav,separated_wav

    def forward_enh(self, wav: torch.Tensor) -> torch.Tensor:
        B,S,T = wav.size()
        wav = torch.flatten(wav, start_dim=0, end_dim=1) 
        shape = jitable_shape(wav)
        wav = _unsqueeze_to_3d(wav)
      
        tf_rep = self.enh_encoder(wav)
        tf_rep = self.enh_enc_activation(tf_rep)
        est_masks = self.enh_masker(tf_rep)
        # print(tf_rep.size(),est_masks.size())
        tf_rep = self.apply_masks(tf_rep, est_masks,sep=True)
        enhanced_wav = self.enh_decoder(tf_rep)
        # print(enhanced_wav.size())
        enhanced_wav = pad_x_to_y(enhanced_wav, wav)
        enhanced_wav = _shape_reconstructed(enhanced_wav, shape)
        # print(enhanced_wav.size())
        enhanced_wav = enhanced_wav.reshape(B,S,T)
        # print(enhanced_wav.size())
        return enhanced_wav
    
    
    def forward_sep(self, wav: torch.Tensor) -> torch.Tensor:
        # print(wav.size())
        shape = jitable_shape(wav)
        wav = _unsqueeze_to_3d(wav)
        # print(wav.size())
        tf_rep_sep = self.sep_encoder(wav)
        tf_rep_sep = self.sep_enc_activation(tf_rep_sep)
      #   print(tf_rep_sep.size())
        est_masks_sep = self.sep_masker(tf_rep_sep)
        masked_tf_rep = self.apply_masks(tf_rep_sep, est_masks_sep,sep=True)
        enhanced_sep_wav = self.sep_decoder(masked_tf_rep)
        reconstructed = pad_x_to_y(enhanced_sep_wav, wav)
        return _shape_reconstructed(reconstructed, shape)

    def apply_masks(self, tf_rep: torch.Tensor, est_masks: torch.Tensor,sep=True) -> torch.Tensor:
        """Applies masks to time-frequency representation.

        Args:
            tf_rep (torch.Tensor): Time-frequency representation in (batch,
                feat, seq) shape.
            est_masks (torch.Tensor): Estimated masks.

        Returns:
            torch.Tensor: Masked time-frequency representations.
        """
        if sep:
            return est_masks * tf_rep.unsqueeze(1)
        else:
            return est_masks * tf_rep

    def get_model_args(self):
        """Arguments needed to re-instantiate the model."""
        fb_config = self.enh_encoder.filterbank.get_config()
        masknet_enh_config = self.enh_masker.get_config()
        masknet_sep_config = self.sep_masker.get_config()
        # Assert both dict are disjoint
        if not all(k not in fb_config for k in masknet_enh_config):
            raise AssertionError(
                "Filterbank and Mask network config share common keys. Merging them is not safe."
            )
        # Merge all args under model_args.
        model_args = {
            **fb_config,
            **masknet_enh_config,
            **masknet_sep_config,
            "encoder_activation": self.encoder_activation,
        }
        return model_args

@script_if_tracing
def _shape_reconstructed(reconstructed, size):
    """Reshape `reconstructed` to have same size as `size`

    Args:
        reconstructed (torch.Tensor): Reconstructed waveform
        size (torch.Tensor): Size of desired waveform

    Returns:
        torch.Tensor: Reshaped waveform

    """
    if len(size) == 1:
        return reconstructed.squeeze(0)
    return reconstructed


# Backwards compatibility
BaseTasNet = BaseEncoderMaskerDecoder
