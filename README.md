# Modulated-Spherical-Fourier-Neural-Operator

Modulated-Spherical-Fourier-Neural-Operators (MSFNO) are an extension of Spherical-Fourier-Neural-Operators by [Bonev et al.](https://arxiv.org/abs/2306.03838)[1]. MSFNO introduces a linear modulation to the feature maps in a SFNO architecture to enable efficient fine-tuning.
The modulation can be applied to pretrained models and conditioned on new training data allowing a model to be fine-tuned to new tasks while having access to additional, task-relevant data.
The performance of MSFNO was evaluated by improving long range weather forecast skill (also called S2S-skill: Subseasonal-to-Seasonal) of an SFNO network.

<p align="center">
  <img src="/figures/RSME_2m_temperature_MSFNO.gif">
  <em align="center"> The figure shows the RMSE for the 2m-temperature field forecasted with MSFNO for a month long rollout </em>
</p>

## Installation

All required packages to run this repository can be installed with conda from the conda_environment.yml file: ```conda install -f conda_environment.yml```
MSFNO is trained on a subset of the ERA5 dataset. It can be downloaded from Weatherbench2 by
```
gsutil -m cp -n -r "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
```
You can prune the dataset down to only store the required data ([prune_weatherbench_ERA5.sh](/data_process/prune_weatherbench_ERA5.sh)).
MFSNO needs the additional variables u100 and v100 (wind velocities in 100m) which are not part of Weatherbench2 and need to be downloaded speratly from ECMWFs Copernicus with the [download_relative_humidity.py](/data_process/download_relative_humidity.py) script.

The weights for a pretrained SFNO network (provided by ECMWF) can be downloaded using
```
python main.py --download-weights
```

## More about MSFNO
![Sketch of the MSFNO architecture](/figures/MSFNO_Architecture.png)
MSFNO can be described in 4 subdivisions. The SFNO-Network predicts the next weather state autoregressively in 6 hour increments. The input data is encoded into tokens by an encoder. The SFNO network as a whole can be roughly understood as operating like a transformer network. 

It is comprised out of 12 SFNO-Blocks which utilize fourier transforms and a learned kernel matrix multiplied in frequency space to perform a global convolution and spacial token mixing. The core insight is that a multiplication in frequency space equals a convolution in normal space, which allows for cheaper computation of global convolutions in $\mathcal{O}(n\log{}n)$ time. For more information on SFNO see [Bonev et al.](https://arxiv.org/abs/2306.03838)[1]

A FiLM-Layer is introduced before the channel mixing MLP of the SFNO Block. This **F**eaturew**i**se **L**inear **M**odulation is described in detail by [Perez et al.](https://arxiv.org/abs/1709.07871) [2]. FiLM-layers influence neural network computation via a simple, feature-wise affine transformation: $\operatorname{FiLM}\left(\boldsymbol{F}_{i, c} \mid \gamma_{i, c}, \beta_{i, c}\right)$= $(1+\gamma_{i,c})$ $\boldsymbol{F}_{i,c}$+$\beta_{i,c}$, where the Features $\boldsymbol{F}_{i, c}$ are transformed by the FiLM-Parameters $\gamma$ and $\beta$.

The FiLM-Parameters are computed by the FiLM-Generator, which can utilize new, out-of-distribution data to modulate the original SFNO network. For each FiLM-layer and each feature dimension a scaling parameter $\gamma$ and shifting parameter $\beta$ is computed by the FiLM-Generator.

## MFSNO long-term weather forecast

<p align="center" widht="100%">
  <p align="center" widht="100%">
    <img src="/figures/SFNO_per_Variable_MSE.png" width="49%"/>
    <img src="/figures/MSFNO_per_Variable_MSE.png" width="49%"/> 
  </p>
  <em align="center"> These figures show the MSE for each atmospheric variable used in SFNO for forecast times of 6 hours nd 7,14,21 and 28 days. On the left side the MSE for the original SFNO is depicted, on the right side the MSE for MSFNO. </em>
</p>

<p align="center">
  <img src="/figures/FiLM_parameters.png">
  <em align="center"> This figure shows a histogram of the FiLM-Parameters which  </em>
</p>

## References

[1] Bonev B., Kurth T., Hundt C., Pathak, J., Baust M., Kashinath K., Anandkumar A.; Spherical Fourier Neural Operators: Learning Stable Dynamics on the Sphere; arXiv 2306.0383, 2023.

[2] Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin, Aaron Courville; FiLM: Visual Reasoning with a General Conditioning Layer, arXiv 1709.07871, 2017.
