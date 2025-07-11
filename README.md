# Chronos-Sea-Level-and-Anomaly-Detection-Extensions

## Overview

This project presents two original extensions of the [Chronos](https://github.com/amazon-science/chronos-forecasting) time series foundation model, demonstrating its flexibility beyond standard forecasting tasks. Specifically:

- **First Extension**: A zero-shot, lightweight anomaly detection method using Chronos-T5-Tiny, evaluated on real-world benchmark datasets (NAB).
- **Second Extension**: A fine-tuning pipeline for adapting Chronos to the domain of sea level forecasting, tested on satellite-derived datasets provided by NOAA.

The experiments are designed to explore Chronos’ performance in both unsupervised anomaly detection and domain-specialized forecasting tasks.

---

## Methodology

### First Extension: Zero-Shot Anomaly Detection

Chronos-T5-Tiny is used without any fine-tuning to forecast time series windows. Anomaly scores are computed from the deviation of real values outside the predicted quantile intervals \([q_\text{low}, q_\text{high}]\). These scores are normalized, smoothed, and thresholded to detect anomalies.

**Key features:**
- No training or fine-tuning required
- Sliding window with stride for computational efficiency
- Quantile-based scoring
- Analysis with NAB real-world datasets

### Second Extension: Fine-Tuning for Sea Level Forecasting

This extension investigates the benefit of fine-tuning the Chronos model on satellite-based sea level datasets from NOAA (e.g., TOPEX/Poseidon, Jason missions). The fine-tuned model is evaluated on both in-domain and zero-shot forecasting tasks.

**Key features:**
- Curated sea level datasets with seasonal and spatial variations
- Training vs zero-shot comparison
- Copernicus and NOAA

---

## Usage

### Requirements
**Main packages:**

- [`torch`](https://pytorch.org/): for running and fine-tuning deep learning models
- [`pandas`](https://pandas.pydata.org/): for time series data manipulation
- [`matplotlib`](https://matplotlib.org/): for plotting time series and anomalies
- [`tqdm`](https://tqdm.github.io/): for progress bars during iterations
- [`scikit-learn`](https://scikit-learn.org/): for evaluation metrics (e.g., precision, recall, F1)
- [`scipy`](https://scipy.org/): for smoothing and statistical operations
- [`chronos`](https://github.com/amazon-science/chronos-forecasting): the base forecasting model used in both extensions

### Running the Notebooks

#### First Extension – Anomaly Detection

1. Download the NAB dataset (e.g., `realKnownCause`) and place the CSV files inside: /anomaly_data
2. Open the notebook: notebook anomaly.ipynb
This notebook will compute anomaly scores using Chronos-T5-Tiny in a zero-shot setting and evaluate them using thresholding methods.

> This notebook computes anomaly scores using Chronos-T5-Tiny in a zero-shot setting and evaluates them via different thresholding strategies.

---

#### Second Extension – Sea Level Forecasting

1. Place your sea level datasets (e.g., NOAA CSVs) inside the following folder: /training_data
2. Launch the notebook: sea_level.ipynb for the processing of the data and the evalution of the model
3. Launch the notebook: chronos_finetuning.ipynb to train the model

> This notebook fine-tunes the Chronos model on sea level time series data and evaluates its performance on in-domain and zero-shot tasks.

---

## Results

### Anomaly Detection

The tables below report the performance of our Chronos-based anomaly detection framework on two datasets from the NAB benchmark: **CPU Utilization** and **Rogue Agent Key Hold**. We compare the results of Chronos under two thresholding strategies (Percentile 95% and IQR) with two classical statistical baselines (Z-Score and SMA Residual).

#### Table 1: CPU Utilization Dataset

**Chronos Methods**

| Method         | Precision | Recall | F1-Score |
|----------------|-----------|--------|----------|
| Percentile 95% | 0.970     | 0.151  | 0.261    |
| IQR            | 0.879     | 0.585  | 0.703    |

**Statistical Methods**

| Method       | Precision | Recall | F1-Score |
|--------------|-----------|--------|----------|
| Z-Score      | 0.290     | 0.057  | 0.096    |
| SMA Residual | 0.346     | 0.115  | 0.172    |

#### Table 2: Rogue Agent Key Hold Dataset

**Chronos Methods**

| Method         | Precision | Recall | F1-Score |
|----------------|-----------|--------|----------|
| Percentile 95% | 0.667     | 0.222  | 0.333    |
| IQR            | 0.242     | 0.296  | 0.267    |

**Statistical Methods**

| Method       | Precision | Recall | F1-Score |
|--------------|-----------|--------|----------|
| Z-Score      | 0.379     | 0.058  | 0.100    |
| SMA Residual | 0.303     | 0.226  | 0.259    |

These results confirm that our Chronos-based approach, even in a **zero-shot setting with no fine-tuning**, outperforms traditional methods in terms of F1-Score, particularly on the CPU Utilization dataset. The IQR thresholding strategy, combined with Chronos-T5-Tiny, achieves the highest overall performance.

### Sea Level Adaptation

To evaluate the impact of domain-specific fine-tuning, we compared the performance of three forecasting models on regional sea level datasets:

- **Chronos:** The pretrained foundation model.
- **Chronos Fine-Tuned:** The Chronos model further trained on sea level data.
- **AutoARIMA:** A statistical baseline commonly used in time series forecasting.

We use two key evaluation metrics:

- **MASE (Mean Absolute Scaled Error):** Lower is better; values below 1 indicate performance better than a naive seasonal forecast.
- **WQL (Weighted Quantile Loss):** Evaluates predictive uncertainty; lower values indicate better coverage of confidence intervals.

#### Table: MASE and WQL Comparison Across Models and Seas

| Sea                    | Chronos MASE | Chronos WQL | Fine-Tuned MASE | Fine-Tuned WQL | AutoARIMA MASE | AutoARIMA WQL |
|------------------------|--------------|-------------|------------------|----------------|----------------|----------------|
| ATLANTIC_OCEAN         | 1.122        | 0.177       | 1.017            | 0.148          | 1.783          | 0.110          |
| BALTIC_SEA             | 0.786        | 0.608       | 0.542            | 0.397          | 1.738          | 0.512          |
| GULF_of_AMERICA        | 0.675        | 0.215       | 0.787            | 0.244          | 3.063          | 0.346          |
| ANDAMAN_SEA            | 0.656        | 0.291       | 0.626            | 0.269          | 2.449          | 0.501          |
| BERING_SEA             | 0.863        | 0.370       | 0.733            | 0.277          | 2.229          | 0.390          |
| BAY_OF_BENGALS         | 0.812        | 0.217       | 0.662            | 0.164          | 2.670          | 0.255          |
| INDIAN_OCEAN           | 0.807        | 0.092       | 1.086            | 0.131          | 1.873          | 0.125          |
| ADRIATIC_SEA           | 0.818        | 0.473       | 0.458            | 0.261          | 1.382          | 0.382          |
| ARABIAN_SEA            | 0.515        | 0.197       | 0.873            | 0.365          | 4.563          | 0.441          |
| CARIBBEAN_SEA          | 0.650        | 0.189       | 0.778            | 0.231          | 2.289          | 0.333          |
| SEA_of_OKHOTSK         | 0.901        | 0.301       | 0.609            | 0.199          | 1.403          | 0.254          |
| PERSIAN_GULF           | 1.325        | 0.605       | 0.711            | 0.303          | 1.104          | 0.291          |
| NORTH_SEA              | 0.805        | 0.422       | 0.367            | 0.169          | 1.611          | 0.345          |
| SEA_of_JAPAN           | 0.516        | 0.224       | 0.719            | 0.310          | 2.571          | 0.291          |
| INDONESIAN             | 1.316        | 0.574       | 0.728            | 0.314          | 5.056          | 0.539          |
| PACIFIC_OCEAN          | 0.909        | 0.068       | 0.888            | 0.068          | 0.937          | 0.055          |
| MEDITERRANEAN_SEA      | 0.684        | 0.357       | 0.694            | 0.364          | 2.358          | 0.366          |
| NORTH_PACIFIC_OCEAN    | 0.406        | 0.119       | 0.621            | 0.195          | 3.730          | 0.343          |
| NORTH_ATLANTIC_OCEAN   | 0.671        | 0.238       | 0.682            | 0.253          | 3.144          | 0.400          |
| TROPICS                | 0.803        | 0.043       | 1.106            | 0.058          | 1.575          | 0.044          |
| SOUTH_CHINA_SEA        | 0.965        | 0.409       | 0.653            | 0.261          | 3.412          | 0.477          |
| YELLOW_SEA             | 0.678        | 0.389       | 1.132            | 0.631          | 3.213          | 0.615          |

As observed, fine-tuning Chronos significantly improves **MASE** in most regions, suggesting better adaptation to sea level patterns. While **AutoARIMA** occasionally achieves lower WQL (e.g., ATLANTIC_OCEAN), it often suffers from much higher MASE. The fine-tuned model achieves a good trade-off, outperforming both baselines on several key regions.

---

## Conclusion
Chronos, originally designed for forecasting, proves to be a flexible and effective backbone for both zero-shot anomaly detection and domain-adapted time series forecasting. These extensions provide practical solutions that combine low resource usage with strong generalization, bridging the gap between research models and real-world deployment.
