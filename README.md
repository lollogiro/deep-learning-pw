# Benchmarking Efficient Modern CNNs for Meteorological Image Classification

## Project overview
This project focuses on the automated classification of weather conditions using images sourced from wildlife camera traps. The primary objective is to modernize benchmarks established in recent ecological literature by evaluating the performance of the latest efficient Convolutional Neural Networks (CNNs) against legacy architectures (MobileNetV1 and V2).

## Dataset
The study utilizes the dataset associated with the paper ["Micro-weather classifications from wildlife cameras"](https://onlinelibrary.wiley.com/doi/10.1111/gcb.17078) (available on [Zenodo](https://zenodo.org/records/10137731)).

* **Source:** data collected from 49 wildlife cameras located in the Maloti-Drakensberg (South Africa) and the Swiss Alps.
* **Classes:** the task involves classifying four distinct weather conditions: `overcast`, `sunshine`, `hail`, and `snow`.
* **Challenges:** the dataset presents real-world difficulties such as variable illumination, diverse background environments, and class imbalance.

## Baselines models
The project compares modern "edge-optimized" architectures against the small MobileNetV1 and MobileNetV2 baselines used in the original study. The specific models evaluated are:

* **MobileNetV3**
    * Implementation: [timm/mobilenetv3_small_100.lamb_in1k](https://huggingface.co/timm/mobilenetv3_small_100.lamb_in1k)
* **MobileNetV4**
    * Implementation: [timm/mobilenetv4_conv_small.e2400_r224_in1k](https://huggingface.co/timm/mobilenetv4_conv_small.e2400_r224_in1k)

These models were selected to ensure a fair performance comparison with the similarly sized models trained in the original paper.
