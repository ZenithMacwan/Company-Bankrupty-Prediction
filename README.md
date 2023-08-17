# Bankruptcy Prediction Model

![Bankruptcy Prediction](https://img.shields.io/badge/Bankruptcy%20Prediction-Model%20Project-blue)

This repository contains a bankruptcy prediction model that utilizes machine learning techniques to predict whether a company is at risk of going bankrupt based on various financial and performance metrics. Bankruptcy prediction is a crucial task for investors, financial analysts, and businesses to assess the financial health and stability of companies.

## Table of Contents

- [About](#about)
- [Dataset](#dataset)
- - [Similar Datasets](#similar-datasets)
- [Project Highlights](#project-highlights)
- [Features](#features)
- [Getting Started](#getting-started)
- - [Installation](#installation)
- - [Usage](#usage)
- [Model Evaluation](#model-evaluation)
- [Contributing](#contributing)
- [Source and Relevant Papers](#source-and-relevant-papers)

## About

This project focuses on the development of a predictive model that can assist in identifying companies at risk of bankruptcy. By analyzing a comprehensive set of financial ratios and indicators, the model aims to provide insights into the potential financial distress of a company.

## Dataset

The dataset used for bankruptcy prediction consists of a wide range of financial and performance metrics for companies spanning over multiple years. The dataset is collected from the Taiwan Economic Journal for the years 1999 to 2009. Bankruptcy labels are defined based on the business regulations of the Taiwan Stock Exchange.

## Project Highlights

- Utilized a diverse set of financial ratios and indicators for bankruptcy prediction.
- Applied machine learning algorithms to develop a predictive model.
- Assessed the financial health of companies based on the developed model.
- Provided insights into potential financial distress and bankruptcy risk.

## Features

- Exploratory data analysis (EDA) and data preprocessing.
- Feature selection and engineering to identify relevant predictors.
- Model selection, training, and evaluation.
- Interpretation of model results and feature importance.

## Getting Started

To start using the bankruptcy prediction model, follow these steps:

### Installation

Clone the repository:

```bash
git clone https://github.com/zenithmacwan/Company-Bankruptcy-Prediction.git
cd Company-Bankruptcy-Prediction.git
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Usage

Run the bankruptcy prediction script:

```bash
python predict_bankruptcy.py
```

## Model Evaluation

The bankruptcy prediction model's performance is evaluated using various metrics, including accuracy, precision, recall, F1-score, and ROC-AUC. These metrics provide insights into the model's ability to predict bankruptcy accurately.

## Contributing

Contributions to this project are welcome. If you encounter any issues, have suggestions for improvements, or want to contribute to the model's development, please feel free to open an issue or submit a pull request.

## Source and Relevant Papers

- Data Source: Deron Liang and Chih-Fong Tsai, National Central University, Taiwan.
- [UCI Machine Learning Repository - Taiwanese Bankruptcy Prediction](https://archive.ics.uci.edu/ml/datasets/Taiwanese+Bankruptcy+Prediction)
- Relevant Paper: Liang, D., Lu, C.-C., Tsai, C.-F., and Shih, G.-A. (2016) Financial Ratios and Corporate Governance Indicators in Bankruptcy Prediction: A Comprehensive Study. European Journal of Operational Research, vol. 252, no. 2, pp. 561-572.
- [Link to Paper](https://www.sciencedirect.com/science/article/pii/S0377221716000412)

---

By Zenith Macwan(https://github.com/zenithmacwan)
