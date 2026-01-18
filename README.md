# MLOps Workflow Primer (Local Kubernetes w/ scikit-learn)

This repository is a lightweight, end‑to‑end MLOps / ML‑workflow primer designed to run **entirely on a local machine** using Kubernetes running on Vagrant (VirtualBox). It is a small simulation environment that helps DevOps / SRE engineers learn the blocks of an ML Pipeline and get a introductory primer into what MLOps is. Having it run locally within the confines of one's personal laptop allows for free experimentation without fear of incurring cloud costs and having it run on Kubernetes on VMs gives it a close feel to a DevOps Engineer's comfortable playing field and allows for a peak into what infrastructure issues/concerns an MLOps infrastructure might encounter when deployed full-scale - again, without fear of incurring cloud costs.

As such, the goal is **not** model sophistication, but to demonstrate visually the following blocks of an ML Workflow/Pipeline as simple as possible:

Machine Learning Pipeline

- [Data ingestion from an external source](https://github.com/journeyofthesoul/mlops-primer/tree/feature/implement-mlops?tab=readme-ov-file#data-ingestion-from-an-external-source)
- [Feature Engineering](https://github.com/journeyofthesoul/mlops-primer/tree/feature/implement-mlops?tab=readme-ov-file#feature-engineering)
- [Model Training](https://github.com/journeyofthesoul/mlops-primer/tree/feature/implement-mlops?tab=readme-ov-file#model-training)
- [Model Evaluation](https://github.com/journeyofthesoul/mlops-primer/tree/feature/implement-mlops?tab=readme-ov-file#model-evaluation)
- [Serving the Model to end users (Inference) via API](https://github.com/journeyofthesoul/mlops-primer/tree/feature/implement-mlops?tab=readme-ov-file#serving-the-model-to-end-users-inference-via-api)

MLOps Technology (using k8s and MLFlow)

- [Automation of Ingestion, Experimentation, Training and Evaluation using Python + k8s CronJob](https://github.com/journeyofthesoul/mlops-primer/tree/feature/implement-mlops?tab=readme-ov-file#automation-of-ingestion-experimentation-training-and-evaluation-using-python--k8s-cronjob)
- [Workflow for Model Promotion, Tagging, Aliasing](https://github.com/journeyofthesoul/mlops-primer/tree/feature/implement-mlops?tab=readme-ov-file#workflow-for-model-promotion-tagging-aliasing)
- [Model Persistence and Registry (allows for manual ops, rollbacks to older versions of the Model)](https://github.com/journeyofthesoul/mlops-primer/tree/feature/implement-mlops?tab=readme-ov-file#model-persistence-and-registry-allows-for-manual-ops-rollbacks-to-older-versions-of-the-model)
- [Experiment Tracking](https://github.com/journeyofthesoul/mlops-primer/tree/feature/implement-mlops?tab=readme-ov-file#experiment-tracking)

## Service Description

The API predicts whether the price for the SPDR S&P 500 ETF Trust (Ticker = SPY) would go up or down the following day. With a simple API call, you'll get one of two answers - UP or DOWN. Internally, the model is constantly trained on past data. I should stress, that the ML model used is arbitrary, something quick to set up and help the DevOps engineer see what the blocks of an ML Pipeline is. <ins>The goal, as explained earlier, is not model sophistication so the accuracy is pretty low</ins>.

## Setup Procedure

The command below will set up everything - the entire kubernetes cluster, the jobs, services, mlflow, and bring the API up. Boot-up time still needs to be improved, but everything should be ready 15 minutes after (usually shorter).

```bash
vagrant up
```

To call the API serving the model, run the following curl command (with or without the threshold query parameter):

```bash
% curl http://192.168.56.21:30951/predictionForTomorrow
{"prediction":"DOWN","confidence":0.0,"threshold":0.5,"model_source":"mlflow","model_alias":"champion"}
% curl "http://192.168.56.21:30951/predictionForTomorrow?threshold=0.3"
{"prediction":"DOWN","confidence":0.0,"threshold":0.3,"model_source":"mlflow","model_alias":"champion"}
```

Note: The IP of 192.168.56.21 is fixed, for a one worker node cluster (with one control plane node) which is configured by default in the _Vagrantfile_. Likewise, the port of 30951 had also been configured fixed as the Service NodePort. If you download this repo AsIs and run _vagrant up_, none of these parameters need to be changed.

To bring the service down:

```bash
vagrant destroy
```

Or, to shut down the infrastructure temporarily (shut down the VMs):

```bash
vagrant halt
```

Either way, _vagrant up_ will bring back the entire infrastructure.

---

## Descriptions for each blocks

This section focuses on high-level descriptions for each of the ML / MLOps blocks the projects includes.

### Data ingestion from an external source

Historical daily market data is retrieved from Yahoo Finance using a dedicated data source abstraction (for future extensibility). Right now, it's fixed to get data for one ticker (SPY).

You may notice that the pipeline trains models using data anchored to a point in the past (one year prior to the current run). The original intention of this project was to operate on a daily cadence: ingesting new market data each day, retraining the model with the most recent observations, and generating predictions for the following day.

In practice, this would require the system to run continuously over many days before meaningful model history and experiment tracking could be observed. For the purposes of demonstrating the full machine learning workflow—data loading, feature engineering, training, evaluation, and model versioning—this repository compresses the time scale.

Specifically, a 5-minute cadence is used to simulate a daily training loop. Each run treats the current execution time as a new “day,” pulls data relative to a historical anchor, retrains the model, and evaluates it on a forward-looking window. This allows multiple model versions, evaluations, and promotions to be generated within a short period of time, making the end-to-end pipeline behavior visible without requiring long-running execution.

Conceptually, this mirrors how a real production system would operate on a daily schedule—only accelerated for demonstration and experimentation purposes.

### Feature Engineering

Feature engineering transforms the raw daily price data into a feature set suitable for the algorithm (model) to learn patterns from and perform predictions.

The following features are added to the data using rolling windows:

- **Daily returns**: capture short-term momentum
- **Moving averages (3, 5 days)**: represent trend direction across multiple time scales
- **Rolling volatility (3 days)**: encode recent market uncertainty

The use of these features for stock market prediction is common as seen in articles online:  
https://www.kaggle.com/code/adityasingh01676/stock-price-prediction-random-forest-regression/notebook
https://medium.datadriveninvestor.com/forecasting-real-time-market-volatility-using-python-282e78b61022

Intuitively, it is easy to see that predicting future market behavior should not rely on past prices alone. Market movements are also influenced by broader trends, which moving averages help capture, as well as by volatility, which reflects how stable or unpredictable recent price movements have been.

At this stage, the exact choice of window sizes (such as 5, 10, or 20 days) is not critical. The important idea is that this step of the ML pipeline enriches the raw price data by deriving additional fields that summarize recent behavior. These derived signals provide more informative inputs for the machine learning model to learn patterns from.

This process of creating new, meaningful inputs from existing data is known as _Feature Engineering_.

### Model Training

This project trains a binary classifier which predicts whether the price will go UP (1) or DOWN (0), rather than attempting to predict an exact future price. This makes the problem simpler, more stable, and easier to reason about.

The machine learning algorithm used is Random Forest Classification. At a high level, a model can be thought of as _Algorithm + Hyperparameters + Data_. Training is the process in which the algorithm learns internal rules (numerical coefficients, weights, configurations) from historical data or commonly known as training data (which has been enriched through feature engineering) in order to best match the known labels. The _hyperparameters_ control how complex the model is and how it learns from the data.

Training in this project is implemented using _scikit-learn (sklearn)_, a widely used Python library for supervised machine learning. Sklearn provides reliable, well-tested implementations of algorithms such as Random Forests and is commonly used in tutorials you can easily find online.

For each training run, the project evaluates a small set of predefined hyperparameter combinations. Each combination is treated as a separate _experiment_ and trained in a loop - each with its own set of hyperparameters. Two key hyperparameters - n_estimators, and max_depth, which control the complexity of the Random Forest model are varied for each experiment.

```python
EXPERIMENTS = [
    {"n_estimators": 50, "max_depth": 3},
    {"n_estimators": 100, "max_depth": 5},
    {"n_estimators": 200, "max_depth": 8},
]
```

### Model Evaluation

Each trained model is evaluated on a strictly forward-looking evaluation window to simulate real-world performance. Training and evaluation occur within the same experiment loop, where predictions are made only on data that lies after the training period.  

The resulting accuracy is compared against the currently registered champion model, allowing the pipeline to automatically identify and promote better-performing candidates in a fully repeatable and deterministic manner.  

### Serving the Model to end users (Inference) via API

The inference service exposes a lightweight HTTP API (using FastAPI) that loads the currently promoted champion model from MLflow and serves real-time predictions. When run from within the kubernetes cluster, the model is pulled from the MLFlow service, and when run locally it is loaded from a local path.

Instead of retraining or recalculating features at request time, the API focuses solely on model inference, keeping runtime latency low and system responsibilities clearly separated.

### Automation of Ingestion, Experimentation, Training and Evaluation using Python + k8s CronJob

### Workflow for Model Promotion, Tagging, Aliasing

### Model Persistence and Registry (allows for manual ops, rollbacks to older versions of the Model)

### Experiment Tracking
