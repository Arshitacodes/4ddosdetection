# 4ddosdetection

# DDoS Detection Using Random Forest

This project focuses on detecting **Distributed Denial of Service (DDoS)** attacks in network traffic using the **Random Forest** machine learning model. It uses the **UNSW-NB15** dataset for training and testing the model and evaluates the effectiveness of the model in detecting DDoS attacks with high accuracy.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Evaluation](#model-evaluation)
- [Future Enhancements](#future-enhancements)

## Project Overview
The goal of this project is to build a robust **DDoS detection system** that can accurately classify network traffic as **normal** or **DDoS attack**. Using the **Random Forest** model, the system is trained on the **UNSW-NB15** dataset, which contains both benign and malicious traffic data. The model is evaluated using various performance metrics such as **accuracy**, **precision**, **recall**, **F1-score**, and **confusion matrix**.

## Dataset
The project uses the **UNSW-NB15** dataset, which is specifically designed for network intrusion detection tasks. The dataset contains labeled instances of both **normal** network traffic and multiple types of **DDoS attacks**.

Dataset download link: [UNSW-NB15 Dataset](https://www.unsw.edu.au/about-us/our-story/our-story-portfolio/cyber-security/our-data)

## Requirements
Before running the code, make sure to install the following libraries:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `imblearn`

You can install the dependencies by running:

```bash
pip install -r requirements.txt

