from .vilint import *
from diffusers.training_utils import EMAModel

class ModelEMA(EMAModel):
    def __init__(
        self,
        model_builder,
        update_after_step=0,
        inv_gamma=1.0,
        power=2/3,
        min_value=0.0,
        max_value=0.9999,
        device=None,
    ):
        """
        Instead of deep copying a provided model, we build the EMA model
        using the model_builder callable.
        """
        # Build the model directly
        model = model_builder
        model.eval()
        model.requires_grad_(False)

        # Instead of calling the super().__init__ which does a deepcopy,
        # we manually set up the instance attributes.
        self.averaged_model = model
        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value

        if device is not None:
            self.averaged_model = self.averaged_model.to(device=device)

        self.decay = 0.0
        self.optimization_step = 0
