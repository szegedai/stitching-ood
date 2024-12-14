from .stitching_layer import StitchingLayer
from .conv_to_conv import ConvToConvStitchingLayer
from .trans_to_trans import TransToTransStitchingLayer
from .resized_conv_to_conv import ResizedConvToConvStitchingLayer
from .conv_to_trans import ConvToTransStitchingLayer
from .trans_to_conv import TransToConvStitchingLayer


def get_stitching_layer(stitching_type, **kwargs) -> StitchingLayer:
    if stitching_type == "c2c":
        return ConvToConvStitchingLayer(init=kwargs["init"])
    elif stitching_type == "t2t":
        return TransToTransStitchingLayer(init=kwargs["init"])
    elif stitching_type == "c2t":
        return ConvToTransStitchingLayer(cls_token=kwargs["c2t_cls_token"])
    elif stitching_type == "t2c":
        return TransToConvStitchingLayer(cls_token=kwargs["t2c_cls_token"],
                                         upsample_mode=kwargs["upsample_mode"])
    elif stitching_type == "rc2c_pre":
        return ResizedConvToConvStitchingLayer(resize_type="upsample",
                                               pre_resize=True,
                                               upsample_mode=kwargs["upsample_mode"])
    elif stitching_type == "rc2c_post":
        return ResizedConvToConvStitchingLayer(resize_type="upsample",
                                               pre_resize=False,
                                               upsample_mode=kwargs["upsample_mode"])
