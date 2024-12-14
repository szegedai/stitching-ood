import json
from os import path

import torch
from torch import nn

from general_utils.models import get_model
from general_utils.datasets import DATASET_META


def get_repr_ood_detector(base_folder: str,
                          source_model_path: str,
                          source_layer_name: str
) -> nn.Module:
    """"""

    metadata_file = path.join(base_folder, "meta.json")
    if not path.isfile(metadata_file):
        raise ValueError(f"{metadata_file} not found!")

    metadata = None
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    for model_file, meta in metadata.items():
        if meta["source_model_path"] == source_model_path and meta["source_layer"] == source_layer_name:
            dataset = meta["id_dataset"]
            orig_input_shape = meta["orig_input_shape"]
            required_input_shape = meta["required_input_shape"]
            model_type = meta["model_type"]
            n_classes = DATASET_META[dataset]["n_classes"]

            model_file = path.join(base_folder, model_file)

            model = get_model(model_name=model_type,
                              n_classes=n_classes,
                              n_channels=required_input_shape[0],
                              orig_input_shape=orig_input_shape,
                              repr_input_shape=required_input_shape)

            model.load_state_dict(torch.load(model_file, map_location="cpu"))

            return model

    raise ValueError(f"Representation classifier + OOD detector for model "
                     f"{source_model_path} and layer {source_layer_name} does "
                     f"not exist in {base_folder}!")
