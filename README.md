# AI Studio – Breast Cancer Detection (Sprint 2: MLOps Pipeline)

## Overview
This project focuses on building an AI-based solution for breast cancer detection using histopathology images.  
In Sprint 2, the primary objective was to design and implement an **MLOps pipeline** for reproducible and modular model training using ClearML.

---

## Sprint 2 Objective
- Build an end-to-end MLOps pipeline
- Ensure modularity and reproducibility
- Integrate data loading, preprocessing, and model training
- Track experiments using ClearML

---

## Project Structure
src/
├── data/
│ ├── data_loader.py
│ ├── data_preprocessing.py
│
├── models/
│ ├── train.py
│ ├── evaluate.py
│
├── pipeline/
│ ├── clearml_pipeline.py

---

## MLOps Pipeline (ClearML)

The pipeline consists of three main stages:

1. **Data Loading**
   - Loads image dataset
   - Supports subset sampling for faster testing

2. **Data Preprocessing**
   - Image resizing
   - Normalization
   - Data transformations

3. **Model Training**
   - CNN-based model training
   - Runs on a small subset for quick iteration
   - Tracks training progress via logs

---

## Execution Environment
The pipeline was executed in **AWS SageMaker**, providing a scalable environment for running and testing the workflow.

---

## Experiment Tracking
All runs are tracked using **ClearML**, including:
- Pipeline execution
- Logs for each stage
- Reproducible experiment setup

---

## Key Features
- Modular code structure
- Reproducible pipeline
- Clear separation of concerns
- Scalable design for future improvements

---

## Limitations (Sprint 2 Scope)
- Training performed on a small subset of data
- UI and NLP components were out of scope for this sprint
- Focus was strictly on MLOps pipeline development

---

## Future Work
- Scale pipeline to full dataset
- Integrate advanced models (e.g., YOLO)
- Add NLP module for clinical queries
- Improve automation and deployment

---

## Author
Harshitha Kolgatta Swamy  
Master of Artificial Intelligence – UTS
