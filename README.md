# Modulated-Spherical-Fourier-Neural-Operator

Modulated-Spherical-Fourier-Neural-Operators (MSFNO) are an extension of Spherical-Fourier-Neural-Operators [Bonev et al.](https://arxiv.org/abs/2306.03838) by introducing a linear modulation to the feature maps of a SFNO network.
The modulation can be applied to pretrained models and conditioned on new training data allowing a model to be fine-tuned to new tasks while having access to additional, task-relevant data.
The performance of MSFNO was evaluated by improving long range weather forecast skill (also called S2S-skill: Subseasonal-to-Seasonal) of an SFNO network.

<p align="center">
  <img src="/figures/RSME_2m_temperature_MSFNO.gif">
  <div>RMSE of the 2m-temperature forecast for a month long rollout</div>
</p>

## Installation

All required packages to run this repository can be installed with conda from the conda_environment.yml file: ```conda install -f conda_environment.yml```
MSFNO is trained on a subset of the ERA5 dataset. It can be downloaded from Weatherbench2 by
```
gsutil -m cp -n -r "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
```
You can prune the dataset down to only store the required data ([prune_weatherbench_ERA5.sh](/data_process/prune_weatherbench_ERA5.sh)).
MFSNO needs the additional variables U100 and V100 (wind velocities in 100m) which are not part of Weatherbench2 and need to be downloaded speratly from ECMWFs Coperinucus with the [download_relative_humidity.py](/data_process/download_relative_humidity.py) script.

The weights for a pretrained SFNO network (provided by ECMWF) can be downloaded using
```
python main.py --download-weights
```

## More about MSFNO
![Sketch of the MSFNO architecture](/figures/MSFNO_Architecture.png)
MSFNO can be described in 4 subdivisions. The SFNO-Network predicts the next weather state autoregressively in 6 hour increments. The input data is encoded into tokens by an encoder. The SFNO network as a whole can be roughly be understood as operating like a transformer network. 

It is comprised out of 12 SFNO-Blocks which utilize fourier transforms and a learned kernel matrix multiplied in frequency space to perform a global convolution and spacial token mixing. The insight here is that a multiplication in frequency space equals a convolution in normal space, which allows for cheaper computation of global convolutions in $\mathcal{O}(n\log{}n)$ time.

A FiLM-Layer is introduced before the channel mixing MLP of the SFNO Block. This **F**eaturew**i**se **L**inear **Modulation** is described in detail by 

## MFSNO long-term weather forecast



## References

`ai-models-fourcastnet` is an [ai-models](https://github.com/ecmwf-lab/ai-models) plugin to run [NVIDIA's FourCastNet](https://github.com/NVlabs/FourCastNet).

FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operators
https://arxiv.org/abs/2202.11214

The FourCastNet code was developed by the authors of the preprint: Jaideep Pathak, Shashank Subramanian, Peter Harrington, Sanjeev Raja, Ashesh Chattopadhyay, Morteza Mardani, Thorsten Kurth, David Hall, Zongyi Li, Kamyar Azizzadenesheli, Pedram Hassanzadeh, Karthik Kashinath, Animashree Anandkumar.

Version 0.1 of FourCastNet is used as default in ai-models.
https://portal.nersc.gov/project/m4134/FCN_weights_v0.1/

FourCastNet is released under **BSD 3-Clause License**, see [LICENSE_fourcastnet](LICENSE_fourcastnet) for more details.
