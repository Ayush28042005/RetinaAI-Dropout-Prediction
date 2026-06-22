# RetinaAI – Student Dropout Risk Prediction

## 1. Project Overview

**Project Name:** RetinaAI - Student Dropout Risk Prediction

**One-Liner:** A multimodal machine learning system that predicts student dropout risk by combining academic records, attendance time-series behavior, and counsellor notes.

### Problem Statement

Student dropout is a major challenge for educational institutions. Early indicators — declining grades, irregular attendance, and concerning counsellor remarks — are often scattered across separate systems, which delays intervention.

This project combines three data modalities into a single pipeline that classifies each student into:

- 🟢 Low Risk (0)
- 🟠 Medium Risk (1)
- 🔴 High Risk (2)

This enables faculty and counsellors to identify and support at-risk students earlier.



## 2. Technical Architecture

### Cloud Provider
Kaggle Notebooks (model development & training)

### Frontend
Not applicable (model development project)

### Backend
Python

### Machine Learning Stack
- LightGBM (Gradient Boosted Decision Trees)
- Scikit-learn (preprocessing, cross-validation, metrics)
- Pandas / NumPy

### NLP Stack
- Keyword-based feature extraction from counsellor notes
- Label encoding of note categories

### Database
Competition dataset provided by organizers (tabular academic data, attendance time-series, counsellor notes)



## 3. Project Pipeline

### Step 1: Data Sources

#### Tabular Data
- Branch, Gender, Hostel Status
- Family Income, Parent Education
- Scholarship Status, Part-time Job
- Screen Time, Commute Time
- CGPA per semester (1–4), Backlogs per semester

#### Attendance Time-Series
- Weekly attendance percentage by subject and semester

#### Counsellor Notes (Text)
- Short unstructured counselling observations per student


### Step 2: Feature Engineering

#### Academic Features
- CGPA mean, standard deviation
- CGPA trend (semester 4 − semester 1)
- CGPA minimum

#### Backlog Features
- Total backlogs across semesters
- Backlog trend

#### Attendance Features
- Overall mean / std / min / max attendance
- Per-semester mean attendance
- Per-subject mean attendance
- Attendance trend (vectorized closed-form linear regression slope)
- Most recent semester attendance

#### Counsellor Note Features
- Label-encoded note category (10 unique recurring note phrases)
- Binary keyword flags: stress, dropout mention, demotivated, backlog, struggling, financial, tutoring, positive, no action needed, monitor


### Step 3: Model Training

A single **LightGBM multiclass classifier** was trained using **Stratified 5-Fold Cross-Validation** with early stopping, on the full 45-feature merged dataset.

This approach was chosen for speed and robustness given the hackathon time constraint, prioritizing a clean, reproducible, single-model pipeline over a heavier multi-model ensemble.



## 4. Model Performance

### Validation Performance (Out-of-Fold, 5-Fold Stratified CV)

| Metric         | Score      |
|----------------|------------|
| Accuracy       | **0.7653** |
| Macro F1 Score | **0.7005** |

### Evaluation Metrics

Both **Accuracy** and **Macro F1 Score** were tracked. Macro F1 was given particular attention because the target classes are imbalanced (~60% Low, ~25% Medium, ~15% High), and Macro F1 weights all three classes equally, giving a fairer picture of minority-class (Medium/High risk) performance than raw accuracy alone.

See `reports/confusion_matrix.png` and `reports/feature_importance.png` for visualizations.



## 5. Key Insights

Feature importance analysis showed the strongest predictors of dropout risk were:

- CGPA trend and CGPA mean
- Attendance trend and recent-semester attendance
- Total backlogs
- Counsellor note keywords (stress, dropout mention, demotivated)
- Family income and parent education

Combining attendance trend (not just raw averages) and counsellor-note keyword signals meaningfully improved separation between Medium and High risk classes versus academic features alone.



## 6. Challenges Faced

- Efficiently computing per-student attendance trend (slope) across ~15,000 students without slow row-by-row `.apply()` calls — solved using a vectorized closed-form linear regression.
- Handling a small, repetitive set of counsellor note phrases without overfitting to exact text.
- Balancing performance across all three risk categories given class imbalance.
- Working within a strict one-hour submission deadline.



## 7. What We Learned

- Practical experience in multimodal feature engineering (tabular + time-series + text).
- The importance of vectorized operations over `.apply()` for performance at scale.
- Why Macro F1 is a more meaningful metric than accuracy under class imbalance.
- The value of Stratified K-Fold Cross-Validation for reliable, leakage-free evaluation on imbalanced multi-class problems.



## 8. Future Scope

- Sentence-embedding or transformer-based representations of counsellor notes (vs. keyword flags)
- LSTM/GRU sequence modeling directly on the raw attendance time-series
- Ensembling LightGBM with CatBoost/XGBoost
- Threshold/probability calibration to further optimize Macro F1
- Real-time dashboard for faculty/counsellor use
- Integration with institutional LMS systems



## 9. Proof of Zero-Cost Usage

### Free Resources Used
- Kaggle Notebooks
- Kaggle Dataset Storage
- Open-source ML libraries: LightGBM, Scikit-learn, Pandas, NumPy, Matplotlib

### Scalability Approach
The pipeline is lightweight and model-agnostic; it can be containerized and deployed as a scheduled batch job or on-demand API for institutional student data.



## 10. Important Links

### GitHub Repository
https://github.com/Ayush28042005/RetinaAI-Dropout-Prediction

### Competition
[RETINA AI: Predict Student Dropout Risk with Deep Learning](https://www.kaggle.com/competitions/retina-ai-predict-student-dropout-risk-with-deep-learning)

### Competition Submission
`submission_v1.csv`



## Repository Structure

```
`train_model.py` : full training pipeline with feature engineering and diagrams
`submission_v1.csv` : final competition predictions
`Student_Dropout_Prediction_Presentation.pptx` : project presentation
`.png` : workflow, architecture, confusion matrix and feature importance diagrams
```


## Author

Ayush Saini
