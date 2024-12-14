# stitching-ood


**How not to Stitch Representations to Measure Similarity: Task Loss Matching versus Direct Matching**</br>
András Balogh, Márk Jelasity</br>
AAAI 2025

This repository contains the code to recreate all our experiments and reproduce our results.

## First steps

Create an environment using the `environment.yml` descriptor:
```sh
conda env create -f environment.yml
```

Download datasets:
- CIFAR-10: managed by PyTorch
- SVHN: managed by PyTorch
- ImageNet-1k: download from [here](https://www.image-net.org/download.php). Rename the subfolders in the root directory to `[root]/training_data` and `[root]/validation_data` accordingly, `[root]` will be the access point
- CIFAR-5M: download from [here](https://github.com/preetum/cifar5m) and place in a folder named `[root]/cifar5m`, `[root]` will be the access point
- 300K random images: download from [here](https://github.com/hendrycks/outlier-exposure) and extract to a folder named `[root]/oe`, `[root]` will be the access point

## Scripts


### Model training
`exec_train.py` is responsible for training base models. Its command line arguments are the following:
- `model_type`: model type indicator e.g. `resnet18`; *required*
- `-gpu`: GPU ID, -1 for execution on CPU; default: 0
- `-d`: name of the dataset; default: cifar10
- `-data`: dataset access point (`[root]` in the above examples); default: ./data/pytorch
- `-o`: name of the optimizer; default: adam
- `-b`: batch size; default: 64
- `-e`: number of epochs; default: 100
- `-lr`: initial learning rate; default: 1e-3
- `-wd`: weight decay; default: 1e-5
- `-s`: random seed; default: 0
- `-sched`: scheduler name (not used in the paper)
- `-rlre`: decay LR by 0.1 after these epochs; default: [] (meaning no lr decay)
- `-workers`: number of workers; default: 1
- `-prefetch`: prefetch factor; default: 1
- `--grad-clip`: $\ell_2$-norm gradient clipping threshold; default: None (meaning no gradient clipping)
- `-dir`: save directory; *required*
- `-sf`: save frequency; default: 0

### Model evaluation
`exec_eval.py` is responsible for evaluating base models. Its command line arguments are the following:
- `model_path`: path to the model file; *required*
- `model_type`: model type indicator e.g. `resnet18`; *required*
- `-d`: name of the dataset; default: cifar10
- `-data`: dataset access point (`[root]` in the above examples); default: ./data/pytorch
- `-b`: batch size; default: 64
- `-gpu`: GPU ID, -1 for execution on CPU; default: 0

### Stitching
`exec_stitch.py` is responsible for running a stitching experiment between a pair of layers. Its command line arguments are the following:
- `front_model_path`: path to the front model ($f$ in the paper); *required*
- `end_model_path`: path to the end model ($g$ in the paper); *required*
- `front_model_type`: type indicator for the front model; *required*
- `end_model_type`: type indicator for the end model; *required*
- `front_layer`: the *name* of the source layer; *required*
- `end_layer`: the *name* of the target layer; *required*
- `stitching_type`: stitching layer type; *required*, options that were used in the paper:
    - `rc2c_pre`: stitching between convolutional layers
    - `t2t`: stitching between transformers
- `-i`: initialization; default: rand, options:
    - `rand`: random initialization of the stitching layer
    - `pinv`: direct matching initialization
    - `eye`: identity initialization
- `-pinv-samples`: number of samples for direct matching ($K$ in the paper); default: 100
- `--mod-only-stitch`: if true, only the stitching layer's parameters will be modified during training; **we always set this to true**
- `-upsample`: name of the resizing stategy if spatial dimensions don't match; default: bilinear
- `--front-stitch-rank`: retained rank of the source representation during training and evaluation for the sensitivity test; default: None (full-rank)
- `--front-pinv-rank`: retained rank of the source representation during direct matching for the sensitivity test; default: None (full-rank)
- `-gpu`: GPU ID, -1 for execution on CPU; default: 0
- `-d`: name of the dataset; default: cifar10
- `-data`: dataset access point (`[root]` in the above examples); default: ./data/pytorch
- `-o`: name of the optimizer; default: adam
- `-b`: batch size; default: 256
- `-e`: number of epochs; default: 30
    - leave on 0 for no training
- `-lr`: initial learning rate; default: 1e-3
- `-wd`: weight decay; default: 1e-5
- `-s`: random seed; default: 0
- `-sched`: scheduler name (not used in the paper)
- `-rlre`: decay LR by 0.1 after these epochs; default: [] (meaning no lr decay)
- `-workers`: number of workers; default: 1
- `-prefetch`: prefetch factor; default: 1
- `--grad-clip`: $\ell_2$-norm gradient clipping threshold; default: None (meaning no gradient clipping)
- `-dir`: save directory; *required*
- `--no-save`: if specified, the stitched model will not be saved (the logs, config and summary will still be saved)

The rest of the parameters are miscellaneous and were not used in the paper.

Example:
```sh
python exec_stitch.py models/test/model_5.pt models/test2/model_5.pt resnet18 resnet18 layer4.1 layer1.0 rc2c_pre -d cifar10 -e 0 -b 512 -i pinv --mod-only-stitch
```

### OOD Detector Training
`exec_representation_classifier.py` is responsible for training an OOD detector for one layer of a network. Its command line arguments are the following:
- `source_model_path`: path to the model; *required*
- `source_model_type`: model type indicator; *required*
- `source_layer_name`: name of the layer; *required*
- `-m`: model type indicator for the representation classifier & OOD detector; *required*, options we used in the paper:
    - `repr_cls_resnet18`
    - `repr_cls_vit_tiny`
- `-gpu`: GPU ID, -1 for execution on CPU; default: 0
- `-s`: random seed; default: 0
- pre-training parameters:
    - `-d`: name of the dataset; default: cifar10
    - `-data`: dataset access point (`[root]` in the above examples); default: ./data/pytorch
    - `-b`: batch size; default: 64
    - `-e`: number of epochs; default: 100
    - `-lr`: initial learning rate; default: 1e-3
    - `-wd`: weight decay; default: 1e-5
    - `-rlre`: decay LR by 0.1 after these epochs; default: [] (meaning no lr decay)
    - `-dir`: save directory; *required*
    - `-sf`: save frequency; default: 0
- fine-tuning parameters:
    - `ftidd`: ID dataset name; *required*
    - `ftiddata`: ID dataset access point; default: ./data/pytorch
    - `ftoodd`: auxiliary OOD dataset name; *required*
    - `ftooddata`: auxiliary OOD dataset access point; default: ./data/pytorch
    - `fe`: number of fine-tuning epochs; default: 20
    - `-m-in`: ID energy marginal; default: -25
    - `-m-out`: OOD energy marginal; default: -7
    - `-score`: OOD detection score for fine-tuning, unused in the paper; default: energy
    - `-fdir`: final saving directory, where a copy of the final OOD detector will be placed along with a descriptor; default: ./models/repr_cls_ood_detectors

Example:
```sh
python exec_train_representation_classifier.py models/test2/model_5.pt resnet18 layer1.0 -m repr_cls_resnet18 -d cifar10 -e 100 -b 512 -ftidd cifar5m -ftoodd outlier_exposure -ftooddata /location/to/300k_random_images/oe -dir models/test_ood -fe 10
```

### OOD evaluation of stitched activations
`exec_eval_stitcher_ood.py` is responsible for evaluating the representations generated by stitchers with previously trained OOD detectors. It works on a set of stitching experiments that are located in the same folder. These stitching experiments are required to have the same setting (same front and end models). Its command line arguments are the following:
- `source_dir`: source directory, containing subdirectories with one stitching experiment each; *required*
- `-gpu`: GPU ID, -1 for execution on CPU; default: 0
- `-d`: name of the dataset; default: cifar10
- `-data`: dataset access point; default: ./data/pytorch
- `-idd`: ID dataset; default: None (meaning the above dataset is the ID dataset)
- `iddata`: root of the ID dataset, only matters if `-idd` is specified; default: ./data/pytorch
- `detectors`: folder where OOD detectors and their descriptors are located (see `-fdir` for `exec_train_representation_classifier.py`); default: ./models/repr_cls_ood_detectors
- `scd`: dataset for sanity checking the OOD detectors (to see if they label examples of this dataset ID or OOD); default: None (meaning OOD detectors will not be sanity checked during evaluation)
- `-scdata`: root of the sanity checking dataset; default: ./data/pytorch

Example:
```sh
python exec_eval_stitcher_ood.py results/stitching_m1_to_m2/ -d cifar10 -b 64 -scd svhn
```

### Linear probing
`exec_probe.py` is responsible for the linear probing of layers. Its command line arguments are the following:
- `model_path`: path to the model; required
- `model_type`: model type indicator; required
- `-l`: name of the layers to probe; required
- `-ranks`: list of ranks to retain when probing, a different probing layer will be trained for every specified rank, all ranks apply to all layers; default: None (only full-rank representations will be probed)
- `-gpu`: GPU ID, -1 for execution on CPU; default: 0
- `-d`: name of the dataset; default: cifar10
- `-data`: dataset access point; default: ./data/pytorch
- `-b`: batch size; default: 64
- `-e`: number of epochs; default: 10
- `-lr`: initial learning rate; default: 1e-3
- `-wd`: weight decay; default: 1e-5
- `-s`: random seed; default: 0
- `-workers`: number of workers; default: 1
- `-prefetch`: prefetch factor; default: 1
- `--verbose`: if set, prints the probing accuracy after every epoch

Example:
```sh
python exec_probe.py models/test2/model_5.pt resnet18 -l layer4.0 layer4.1 -d cifar10 -b 256 -ranks 512 448 384 320 256 192 128 64 32 16 8 4 2 1
```

### Computing Low-Rank Similarity Indices
`exec_compute_low_rank_sim_metrics.py` is responsible for computing the similarities of the representations of two specified layers at different ranks (corresponding to the sensitivity test). It can be used for the specificity test as well by not providing the arguments that control low-rank approximation. Its command line arguments are the following:
- `model1_path`: path to the first model; *required*
- `model2_path`: path to the second model; *required*
- `model1_type`: type of the first model; *required*
- `model2_type`: type of the second model; *required*
- `model1_layer`: layer from the first model; *required*
- `model2_layer`: layer from the second model; *required*
- `-m1r`: retained ranks of the representations of the layer from the first model; default: None (only test for full-rank)
- `-m2r`: retained ranks of the representations of the layer from the second model; default: None (only test for full-rank)
- `-gpu`: GPU ID, -1 for execution on CPU; default: 0
- `-d`: name of the dataset; default: cifar10
- `-data`: dataset access point; default: ./data/pytorch
- `-prefetch`: prefetch factor; default: 1
- `--verbose`: if set, prints the probing accuracy after every epoch

Example:
```sh
python exec_compute_low_rank_sim_metrics.py models/test/model_5.pt models/test2/model_5.pt resnet18 resnet18 layer4.1 layer3.0 -m1r 512 448 -m2r 256 128
```

### ImageNet Training
`misc/imagenet_train_main.py` is the script we used for training ResNets on the ImageNet dataset. Usage:
```sh
python imagenet_train_main.py path/to/imagenet -a tv_resnet18 -b 1024 --gpu gpu_id --seed seed
```

## Citation
coming soon...
