# DM2F-UC-Model
Official code for the DM²F-UC model. This model enhances joint perception from a single IMU by fusing temporal/time-frequency features and quantifying uncertainty.
# DM2F-UC-Model

This repository contains the official PyTorch implementation for the paper: **"DM2F-UC-Model"**(2025).

Our proposed model, DM²F-UC, provides a robust perceptual foundation for embodied agents by enhancing joint perception from a single IMU. It synergistically fuses temporal and time-frequency features and integrates end-to-end uncertainty modeling to enable risk-aware control.


---

## Key Features

-   **Single-Source, Multi-Representation Framework:** Enhances joint perception from a single IMU by synergistically fusing temporal and time-frequency features.
-   **Novel "Align-then-Modulate" Fusion:** Introduces a dynamic fusion mechanism using asymmetric cross-modal attention for semantic alignment and a gated module for adaptive weighting.
-   **Integrated Uncertainty Modeling:** Predicts a full Gaussian distribution (mean and variance), enabling robust risk-awareness for safety-critical control.
-   **Superior Performance:** Achieves a 20% reduction in RMSE while maintaining an R² above 0.95, with highly reliable uncertainty quantification.

---

## Requirements

### Tested Environment
-   **Operating System:** Ubuntu 20.04
-   **Python Version:** 3.10+
-   **Hardware:** 12th Gen Intel Core i9, 64 GB RAM; 2x NVIDIA GeForce RTX 3090 (24 GB)

### Installation
To install all required Python libraries, simply run the following command. This will use the provided `requirements.txt` file to set up the environment correctly.

```bash
pip install -r requirements.txt```

---

## Usage

1.  **Prepare Data:** Place your preprocessed dataset CSV file inside the `imu_data/` directory. The project is pre-configured to work with the dataset mentioned below.

2.  **Configure:** The default settings in `config.json` are configured to replicate the results from the paper. You can modify this file to change hyperparameters, file paths, or target columns.

3.  **Run Training & Evaluation:** To run the full training, evaluation, and cross-validation pipeline, execute the main script from the root directory:

    ```bash
    python main.py
    ```
    All results, including logs, performance metrics, and plots, will be saved to the `results/` directory.

---

## Data Source

The experiments in this study were conducted using a publicly available dataset. We extend our gratitude to the authors for making their data accessible.

-   **Dataset Title:** "A Human Lower-Limb Biomechanics and Wearable Sensors Dataset During Cyclic and Non-Cyclic Activities"
-   **Authors:** Scherpereel, K., Molinaro, D., Inan, O. et al.
-   **Availability:** The dataset is available at the [**Georgia Tech Repository**](https://repository.gatech.edu/entities/publication/20860ffb-71fd-4049-a033-cd0ff308339e/).
-   **License:** The dataset is licensed under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license. This means you are free to share and adapt the data, provided you give appropriate credit to the original authors.

---

## License

The source code in this repository is licensed under the MIT License. Please see the [LICENSE](LICENSE) file for more details.

---

## Citation

If you find our work or this repository useful in your research, please consider citing our paper:

```bibtex

```

## Email:
If you have any questions, please email to: shuxu@mail.ustc.edu.cn
