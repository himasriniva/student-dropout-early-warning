# Results

Held-out test set: 885 students (32.1% dropouts). Decision thresholds tuned on out-of-fold training predictions for recall ≥ 80%.

## Selected model per stage (test set)

| stage      | model                  |   threshold |   roc_auc |   pr_auc |   accuracy |   balanced_accuracy |   precision |   recall |    f1 |   flagged_pct |
|:-----------|:-----------------------|------------:|----------:|---------:|-----------:|--------------------:|------------:|---------:|------:|--------------:|
| enrollment | Logistic Regression    |       0.382 |     0.817 |    0.732 |      0.699 |               0.728 |       0.52  |    0.806 | 0.633 |         0.497 |
| semester_1 | Hist Gradient Boosting |       0.382 |     0.897 |    0.856 |      0.831 |               0.824 |       0.707 |    0.806 | 0.753 |         0.366 |
| semester_2 | Logistic Regression    |       0.549 |     0.914 |    0.88  |      0.866 |               0.848 |       0.785 |    0.799 | 0.792 |         0.327 |

## 5-fold CV comparison (training set)

| stage      | model                  |   cv_roc_auc |   cv_average_precision |   cv_f1 |   cv_balanced_accuracy |   cv_roc_auc_std |
|:-----------|:-----------------------|-------------:|-----------------------:|--------:|-----------------------:|-----------------:|
| enrollment | Logistic Regression    |       0.827  |                 0.742  |  0.6522 |                 0.7443 |           0.0157 |
| enrollment | Decision Tree          |       0.7777 |                 0.665  |  0.6046 |                 0.7051 |           0.0159 |
| enrollment | Random Forest          |       0.8237 |                 0.7377 |  0.6342 |                 0.7314 |           0.0072 |
| enrollment | Hist Gradient Boosting |       0.8154 |                 0.7394 |  0.6465 |                 0.7396 |           0.0114 |
| enrollment | SVM (RBF)              |       0.8217 |                 0.7312 |  0.6316 |                 0.7296 |           0.0132 |
| semester_1 | Logistic Regression    |       0.9048 |                 0.8588 |  0.7699 |                 0.8337 |           0.0117 |
| semester_1 | Decision Tree          |       0.8701 |                 0.8101 |  0.7416 |                 0.8126 |           0.016  |
| semester_1 | Random Forest          |       0.9041 |                 0.8561 |  0.766  |                 0.8239 |           0.0102 |
| semester_1 | Hist Gradient Boosting |       0.9049 |                 0.8601 |  0.7721 |                 0.8311 |           0.0135 |
| semester_1 | SVM (RBF)              |       0.9014 |                 0.8497 |  0.7569 |                 0.8157 |           0.0134 |
| semester_2 | Logistic Regression    |       0.9244 |                 0.8852 |  0.7974 |                 0.8543 |           0.0127 |
| semester_2 | Decision Tree          |       0.8914 |                 0.8187 |  0.7545 |                 0.8226 |           0.0131 |
| semester_2 | Random Forest          |       0.9215 |                 0.8796 |  0.7914 |                 0.8415 |           0.0102 |
| semester_2 | Hist Gradient Boosting |       0.9224 |                 0.8858 |  0.8015 |                 0.8525 |           0.011  |
| semester_2 | SVM (RBF)              |       0.9209 |                 0.8781 |  0.7907 |                 0.8416 |           0.0121 |

## All models on test set

| stage      | model                  |   threshold |   roc_auc |   pr_auc |   accuracy |   balanced_accuracy |   precision |   recall |    f1 |   flagged_pct |
|:-----------|:-----------------------|------------:|----------:|---------:|-----------:|--------------------:|------------:|---------:|------:|--------------:|
| enrollment | Logistic Regression    |       0.382 |     0.817 |    0.732 |      0.699 |               0.728 |       0.52  |    0.806 | 0.633 |         0.497 |
| enrollment | Decision Tree          |       0.39  |     0.772 |    0.652 |      0.666 |               0.695 |       0.487 |    0.778 | 0.599 |         0.513 |
| enrollment | Random Forest          |       0.334 |     0.812 |    0.725 |      0.69  |               0.715 |       0.511 |    0.785 | 0.619 |         0.493 |
| enrollment | Hist Gradient Boosting |       0.295 |     0.813 |    0.736 |      0.67  |               0.711 |       0.492 |    0.824 | 0.616 |         0.538 |
| enrollment | SVM (RBF)              |       0.227 |     0.805 |    0.723 |      0.703 |               0.719 |       0.525 |    0.764 | 0.623 |         0.467 |
| semester_1 | Logistic Regression    |       0.49  |     0.905 |    0.862 |      0.837 |               0.821 |       0.733 |    0.775 | 0.753 |         0.339 |
| semester_1 | Decision Tree          |       0.413 |     0.876 |    0.802 |      0.803 |               0.809 |       0.654 |    0.824 | 0.729 |         0.405 |
| semester_1 | Random Forest          |       0.403 |     0.901 |    0.851 |      0.827 |               0.816 |       0.708 |    0.785 | 0.745 |         0.356 |
| semester_1 | Hist Gradient Boosting |       0.382 |     0.897 |    0.856 |      0.831 |               0.824 |       0.707 |    0.806 | 0.753 |         0.366 |
| semester_1 | SVM (RBF)              |       0.312 |     0.897 |    0.841 |      0.842 |               0.832 |       0.729 |    0.806 | 0.766 |         0.355 |
| semester_2 | Logistic Regression    |       0.549 |     0.914 |    0.88  |      0.866 |               0.848 |       0.785 |    0.799 | 0.792 |         0.327 |
| semester_2 | Decision Tree          |       0.442 |     0.889 |    0.816 |      0.833 |               0.825 |       0.712 |    0.803 | 0.755 |         0.362 |
| semester_2 | Random Forest          |       0.435 |     0.916 |    0.877 |      0.876 |               0.86  |       0.8   |    0.817 | 0.808 |         0.328 |
| semester_2 | Hist Gradient Boosting |       0.476 |     0.912 |    0.879 |      0.864 |               0.849 |       0.779 |    0.806 | 0.792 |         0.332 |
| semester_2 | SVM (RBF)              |       0.416 |     0.909 |    0.862 |      0.864 |               0.846 |       0.785 |    0.796 | 0.79  |         0.325 |

## Risk bands (semester-1 model, test set)

| risk_band   |   Dropout |   Enrolled |   Graduate |   n |   actual_dropout_rate |
|:------------|----------:|-----------:|-----------:|----:|----------------------:|
| High        |       204 |         29 |         14 | 247 |                 0.826 |
| Medium      |        35 |         48 |         33 | 116 |                 0.302 |
| Low         |        45 |         82 |        395 | 522 |                 0.086 |

## Top 10 risk drivers (semester-1 model, permutation importance)

| feature                             |   importance_mean |   importance_std |
|:------------------------------------|------------------:|-----------------:|
| 1st sem approval rate               |            0.1316 |           0.0119 |
| Tuition fees up to date             |            0.0383 |           0.0045 |
| Course                              |            0.0129 |           0.0036 |
| Age at enrollment                   |            0.0124 |           0.0018 |
| Gender                              |            0.0102 |           0.0026 |
| Debtor                              |            0.0061 |           0.0019 |
| Curricular units 1st sem (grade)    |            0.0056 |           0.0029 |
| Scholarship holder                  |            0.0041 |           0.0014 |
| Unemployment rate                   |            0.0039 |           0.002  |
| Curricular units 1st sem (approved) |            0.0026 |           0.0015 |

## 3-class model (Dropout / Enrolled / Graduate, semester-2 features)

```
              precision    recall  f1-score   support

     Dropout     0.8000    0.7324    0.7647       284
    Enrolled     0.4624    0.5409    0.4986       159
    Graduate     0.8428    0.8371    0.8400       442

    accuracy                         0.7503       885
   macro avg     0.7017    0.7035    0.7011       885
weighted avg     0.7607    0.7503    0.7545       885
```
