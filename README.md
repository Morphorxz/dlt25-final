# Learning Rate Scheduler Repository

This repository implements a framework for fitting, predicting, and optimizing learning rate (LR) schedules using multi-power law (MPL) models. It is designed for researchers and practitioners in machine learning optimization, offering tools to predict training loss dynamics and derive optimized LR schedules.

## Results

Below are the updated evaluation metrics and best parameters for the MPL model fitted to datasets for 25M, 100M, and 400M parameter models. The results in our paper are not updated due to the compute cost to rerun all experiments.

## Formulation

- The Multi-Power Law is formulated as follows:

    $L(t) = L_0 + A \cdot (S_1(t)+S_W)^{-\alpha} - LD(t), \quad$ where
    $\quad S_1(t) := \sum_{\tau=1}^{t} \eta_\tau$.

    $LD(t) := B \sum_{k=1}^{t} (\eta_{k-1} - \eta_k) \cdot G(\eta_k^{-\gamma}S_{k}(t))$,
    $\ S_k(t) := \sum_{\tau = k}^{t} \eta_{\tau}$,
     and $G(x) := 1 - (Cx + 1)^{-\beta}$.

### Evaluation Metrics

| Model | $R^2$         | MAE           | RMSE          | PredE         | WorstE        |
|-------|---------------|---------------|---------------|---------------|---------------|
| 25M   | 0.9988        | 0.00376       | 0.00465       | 0.00110       | 0.00409       |
| 100M  | 0.9983        | 0.00435       | 0.00592       | 0.00142       | 0.00583       |
| 400M  | 0.9978        | 0.00484       | 0.00730       | 0.00168       | 0.00995       |

- **$R^2$**: Coefficient of Determination.
- **MAE**: Mean Absolute Error.
- **RMSE**: Root Mean Squared Error.
- **PredE**: Average prediction error (relative).
- **WorstE**: Worst-case prediction error (relative).

### Best Parameters

| Model | $L_0$ | $A$   | $\alpha$ | $B$      | $C$   | $\beta$ | $\gamma$ |
|-------|-------|-------|----------|----------|-------|---------|----------|
| 25M   | 3.040 | 0.525 | 0.508    | 363.788  | 2.066 | 0.583   | 0.641    |
| 100M  | 2.651 | 0.601 | 0.453    | 437.946  | 2.132 | 0.598   | 0.655    |
| 400M  | 2.375 | 0.654 | 0.429    | 523.425  | 2.025 | 0.594   | 0.635    |

<!-- - **Coefficients**: Best parameters for the MPL model  -->

<!-- - **Best Loss**: 
  - 25M: 0.0002786
  - 100M: 0.0002751
  - 400M: 0.0004078 -->

## Features
- **LR Schedulers**: Includes cosine, constant, two-stage, WSD, and WSDLD schedules.
- **Optimization**: Optimizes LR schedules using fitted MPL models with constraints of non-increasing LR.
- **Evaluation**: Provides metrics (e.g., MSE, $R^2$, Huber loss) and visualizations of predicted vs. actual loss.
- **Testing**: Unit tests for key components ensure reliability.
<!-- - **MPL Models**: Two models (`MPL` and `MultiPower`) for predicting loss based on LR schedules. -->
<!-- - **Fitting Pipeline**: Parameter initialization via grid search and fine-tuning with AdamW. -->

## Installation

### Prerequisites
- Python 3.8+
- Git

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/thu-yao-01-luo/MultiPowerLaw.git
   cd MultiPowerLaw
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Dependencies include: `numpy`, `torch`, `scipy`, `matplotlib`, `tqdm`, `sklearn`.

3. (Optional) Use a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Usage

### Running the Main Script
The entry point `main.py` executes the full pipeline: data loading, model fitting, evaluation, and LR schedule optimization. Run it with:
```bash
python main.py --folder_path 400 > logs/400.log
```
- `--folder_path` or `-f`: Specifies the model size (`25`, `100`, or `400`). Default is `400`.

The optimization can be executed separately with best parameters `PARAMS` in `config.py`.
```bash
python main.py -opt_only -f 400
```
- `--opt_only` or `-o`: optimization-only option.


**Outputs**:
<!-- - **Initial Data Plots**: LR and loss curves in `./<model_size>M/` (e.g., `./400M/cosine_24000_lrs.png`). -->
- **Fitted Model Evaluation**: Predicted vs. actual loss plots in `./<model_size>M/fit/` (e.g., `./400M/fit/cosine_24000_mplfit.png`).
- **Optimized LR Schedule**: Saved as `./optimized_schedules/<model_size>.npy` and plotted in `./optimized_schedules/<model_size>.png`.
- **Fitting Process Info**: Print intermediate training losses, gradient norm, and intermediate parameters for fitting process, and detailed evaluation results in final.
- **Optimization Process Info**: Print intermediate training losses, gradient norm and first and last 5 steps of the optimized schedule in the optimization process.

### Running Tests
Unit tests are located in `tests/`. Run them with (under folder `MultiPowerLaw/`):
```bash
python -m tests.test_lrs
python -m python -m tests.test_data_loader -f 400
```
- `--folder_path` or `-f`: Specifies the model size (`25`, `100`, or `400`). Default is `400`.
- **LR schedules visualizaton**: `lrs.png` under folder `MultiPowerLaw/`.
- **Initial Data Plots**: LR and loss curves in `./<model_size>M/lrs/` and `./<model_size>M/loss/` (e.g., `./400M/lrs/cosine_24000_lrs.png`, `./400M/loss/cosine_24000_loss.png`).

## Project Structure
```
lr_scheduler_repo/
├── src/                # Core source code
│   ├── __init__.py     # Marks src as a package
│   ├── config.py       # Constants and configurations
│   ├── data_loader.py  # Data loading and preprocessing
│   ├── lr_schedulers.py# LR scheduler functions
│   ├── models.py       # MPL and MultiPower model definitions
│   ├── fitting.py      # Model fitting logic
│   ├── evaluation.py   # Evaluation and plotting functions
│   ├── optimization.py # LR schedule optimization
│   └── utils.py        # Utility functions
├── tests/              # Unit tests
│   ├── __init__.py     # Marks tests as a package
│   ├── test_lr_schedulers.py
│   ├── test_data_loader.py
│   ├── test_models.py
│   └── test_fitting.py
├── main.py             # Entry point script
├── requirements.txt    # Dependencies
└── README.md           # This file
```

### Key Components
- **`config.py`**: Defines datasets, file paths, and constants (e.g., `OPT_PATH` for optimized schedules).
- **`lr_schedulers.py`**: Implements various LR schedules.
- **`models.py`**: Contains `MPL` (primary model for fitting and optimization) and `MultiPower`.
- **`fitting.py`**: Fits the MPL model to training data.
- **`optimization.py`**: Optimizes LR schedules using the fitted MPL model.
- **`evaluation.py`**: Evaluates model performance with metrics and plots.

## Data Requirements
- **Format**: CSV files with columns `step`, `col1`, `loss` (e.g., `0,1,2.0`).
- **Location**: Specified in `FOLDER_PATHS` (e.g., `/home/kairong/lr_drop_law/csv_400_converted/`).
- **Names**: Must match `TRAIN_SET` and `TEST_SET` (e.g., `cosine_24000.csv`).

## Customization
- **New Schedulers**: Add to `lr_schedulers.py` and update `data_loader.py`.
- **Model Adjustments**: Modify `models.py` for different formulations.
- **Hyperparameters**: Tune fitting (`mpl_adam_fit`) or optimization (`optimize_lr_schedule`) parameters in `main.py`.

## Contributing
1. Fork the repo.
2. Create a feature branch: `git checkout -b feature/your-feature`.
3. Commit changes: `git commit -m "Add your feature"`.
4. Push: `git push origin feature/your-feature`.
5. Open a pull request.

Include tests and update documentation for new features.

## License
MIT License (to be added in a `LICENSE` file).

## Acknowledgments
- Built for optimization research in deep learning.
- Powered by PyTorch, NumPy, and other open-source libraries.

## Contact
For issues or questions, open a GitHub issue or email `<your-email>`.

Explore and optimize your learning rate schedules with ease!
