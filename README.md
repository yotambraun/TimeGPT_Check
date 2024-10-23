# TimeGPT_Check

Welcome to the **TimeGPT_Check** repository. This project utilizes [Nixtla's TimeGPT](https://github.com/Nixtla/nixtla) and [HierarchicalForecast](https://github.com/Nixtla/hierarchicalforecast) libraries for time series forecasting and anomaly detection. The aim of this project is to demonstrate how to work with TimeGPT on time series data.

## Project Overview

### TimeGPT
[TimeGPT](https://docs.nixtla.io/) is a generative pre-trained transformer model for time series forecasting. It can predict various domains such as retail, electricity, finance, and IoT. TimeGPT allows you to implement accurate predictions with minimal code, enabling organizations to forecast without needing a specialized machine learning team.

Key features include:
- **Flexible Forecasting**: Predicts across various industries and domains.
- **Ease of Use**: Integrates easily with Python for quick forecasting experiments.
- **Anomaly Detection**: Detects anomalies in time series data efficiently.

### Hierarchical Forecast
[HierarchicalForecast](https://nixtlaverse.nixtla.io/hierarchicalforecast/index.html) is a Python-based library designed for reconciling large collections of time series that follow a hierarchical structure. It offers a variety of reconciliation methods to ensure that the forecasts at different levels of aggregation maintain coherence.

Reconciliation methods include:
- **Classic Reconciliation**:
  - **BottomUp**: Adds lower-level forecasts to upper levels.
  - **TopDown**: Distributes top-level forecasts through hierarchies.
- **Advanced Reconciliation**:
  - **MiddleOut**: Combines both BottomUp and TopDown approaches.
  - **MinTrace**: Minimizes total forecast variance for coherent forecasts.
  - **ERM (Elastic Reconciliation Method)**: Optimizes reconciliation by minimizing an L1 regularized objective.
- **Probabilistic Methods**:
  - **Normality**: Uses MinTrace variance-covariance matrix assuming normality.
  - **Bootstrap**: Generates reconciled predictions using bootstrap methods.
  - **PERMBU**: Reconciles sample predictions by reinjecting multivariate dependencies.

## API Key Setup

Before using TimeGPT, you need to sign up on the [Nixtla Platform](https://nixtla.io/) and obtain an API key. Follow these steps:

1. Go to the [Nixtla api_keys page](https://dashboard.nixtla.io/api_keys) and create an account.
2. Once logged in, navigate to your dashboard and generate an API key.
3. Save your API key securely.

## Experiments and Workflow

The project is organized into several experiments, each focusing on different aspects of time series forecasting and reconciliation.

### Folders and Structure:
- **`more_examples`**: Contains various examples of how to utilize TimeGPT and HierarchicalForecast for forecasting and reconciliation.
- **`preprocess_data`**: Contains utility functions for preprocessing the time series data before forecasting.
- **`fine_tune`**: Contains examples and utilities for fine-tuning the TimeGPT model for improved forecasting accuracy.
- **`hierarchical_forecasting`**: Contains examples of hierarchical reconciliation using methods like BottomUp, MinTrace, and ERM.
- **`cross_validation`**: Demonstrates how to perform cross-validation with TimeGPT for time series models.
- **`electricity_example`**: Demonstrates electricity demand forecasting using TimeGPT.

### Key Files:
- **`preprocess_data/useful_functions.py`**: Provides helper functions to preprocess the data by expanding and aggregating based on frequency.
  - `expand_data_with_zeros`: Expands a time series with zeros where there is no data.
  - `aggregate_by_frequency`: Aggregates time series by a given frequency.

- **`hierarchical_forecasting/hierarchical_forecasting.py`**: Main script for running hierarchical forecasts and reconciliation.
  - Uses reconciliation methods like BottomUp, MinTrace, and ERM.

- **`fine_tune/fine_tune.py`**: Demonstrates how to fine-tune TimeGPT for time series forecasting tasks using MAE (Mean Absolute Error) as the loss function.

- **`cross_validation/cross_validation.py`**: Demonstrates how to perform cross-validation on time series models using TimeGPT, calculating MAPE (Mean Absolute Percentage Error) across different forecast windows.
  - `perform_cross_validation`: Runs cross-validation using TimeGPT and calculates the performance of each window.
  
- **`electricity_example/example_electricity.py`**: Shows how to perform electricity demand forecasting using TimeGPT with visualizations for predicted vs. actual data.
  - `plot_and_save_forecast`: Plots the actual vs forecasted values for each unique electricity demand series.

## Links to Documentation

- [Nixtla's TimeGPT Documentation](https://docs.nixtla.io/)
- [Nixtla's HierarchicalForecast Documentation](https://nixtlaverse.nixtla.io/hierarchicalforecast/index.html)

## Conclusion
This project demonstrates how to use Nixtla's TimeGPT and HierarchicalForecast for advanced time series forecasting tasks. With these tools, you can implement accurate and coherent forecasts across hierarchical structures in just a few lines of code.
Happy Forecasting!