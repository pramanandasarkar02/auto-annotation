# from .transunet3d import create_transunet3d
from .segresnet3d import create_segresnet3d
from .swinunetr3d import create_swin_unetr3d
from .vnet3d import create_vnet3d
from .unetr3d import create_unetr3d
from .unet3d import create_unet3d
# from .attentionunet3d import create_attention_unet3d
# from .custommodel import create_femur_segmentation_model
from .csa3d import create_csa_network




def create_qct_segmentation_model(
    name: str,
    **kwargs,
):
    """
    name ∈ {"unet", "transunet", "unetr", "swinunetr", "segresnet", "attnunet", "vnet"}
    kwargs are forwarded to the corresponding builder above.
    """
    name = name.lower()
    # print("here")
    if name == "unet":
        return create_unet3d(**kwargs)
    # if name == "transunet":
    #     return create_transunet3d(**kwargs)
    if name == "unetr":
        return create_unetr3d(**kwargs)
    if name == "swinunetr":
        return create_swin_unetr3d(**kwargs)
    # if name == "segresnet":
    #     return create_segresnet3d(**kwargs)
    # if name == "vnet":
    #     return create_vnet3d(**kwargs)
    # if name == "attentionunet":
    #     return create_attention_unet3d(**kwargs)
    # if name == "custom":
    #     return create_femur_segmentation_model(**kwargs)
    if name == "csa":
        return create_csa_network(**kwargs)
    raise ValueError(f"Unknown model '{name}'")