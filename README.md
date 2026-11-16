# Predicting Student Mental Health Score

A regression project that predicts a student's `Mental_Health_Score` (a continuous score from about 3.6 to 9.4) from social media usage, sleep, study time, physical activity, stress level and basic demographics.

**Best model:** tuned XGBoost, **test R² = 0.905**, RMSE = 0.412, MAE = 0.302, MAPE = 5.07%.

## Dataset

- File: `Student Social Media And Mental Health Impact.csv`
- 5,000 students, 13 columns (4,998 rows after removing 2 duplicates)
- No missing values
- Source: *add the link and licence here*

| Type | Columns |
|---|---|
| Numeric | Age, Avg_Daily_Usage_Hours, Daily_Unlocks, Study_Hours, Physical_Activity_Hours, Sleep_Hours_Per_Night |
| Ordinal | Stress_Level (Low < Medium < High < Very High) |
| Categorical | Gender, Academic_Level, Most_Used_Platform, Purpose_Of_Use, Country |
| Target | Mental_Health_Score |

## Method

1. **Cleaning**: dropped 2 duplicate rows and clipped impossible negative `Physical_Activity_Hours` values to 0.
2. **EDA**: target distribution, correlation heatmap, and relationships of stress, usage and sleep with the score.
3. **Split first**: 80/20 train/test split before any fitting, to avoid leakage.
4. **Country grouping**: the top 10 most frequent values are computed on the training set only; all other countries become `Other`.
5. **Preprocessing** (one `ColumnTransformer`, bundled inside each model pipeline):
   - `Study_Hours`: median impute, log1p, standard scale
   - other numeric: median impute, standard scale
   - `Stress_Level`: ordinal encoding
   - other categoricals: one-hot encoding (unknown categories ignored)
6. **Models**: Linear Regression, Random Forest, XGBoost, HistGradientBoosting.
7. **Validation**: 5-fold cross-validation on the training set, then `RandomizedSearchCV` for the three tree models.
8. **Final check**: tuned models scored once on the untouched test set.
9. **Interpretation**: permutation importance on the test set.

## Results

**Cross-validation on the training set (default parameters)**

| Model | R² | RMSE | MAE | MAPE |
|---|---|---|---|---|
| Random Forest | 0.8650 | 0.4637 | 0.3432 | 5.82% |
| XGBoost | 0.8629 | 0.4671 | 0.3488 | 5.90% |
| HistGradientBoosting | 0.8402 | 0.5044 | 0.3873 | 6.57% |
| Linear Regression | 0.7206 | 0.6673 | 0.5257 | 9.00% |

**After tuning (CV on train) and on the held-out test set**

| Model | CV R² | Test R² | Test RMSE | Test MAE | Test MAPE | Train R² |
|---|---|---|---|---|---|---|
| **XGBoost** | 0.8848 | **0.9047** | 0.4124 | 0.3023 | 5.07% | 0.9991 |
| HistGradientBoosting | 0.8803 | 0.8967 | 0.4293 | 0.3121 | 5.26% | 0.9910 |
| Random Forest | 0.8718 | 0.8949 | 0.4332 | 0.3210 | 5.41% | 0.9842 |

XGBoost was the best model in both cross-validation and on the test set, so it was chosen on validation results and confirmed on test. Tuning improved every model over its default score.

## What drives the predictions

Permutation importance (drop in test R² when a feature is shuffled), top features for XGBoost:

| Feature | Importance |
|---|---|
| Avg_Daily_Usage_Hours | 0.655 |
| Sleep_Hours_Per_Night | 0.193 |
| Daily_Unlocks | 0.124 |
| Most_Used_Platform | 0.059 |
| Country | 0.046 |

Usage and sleep carry most of the predictive signal. `Stress_Level` ranks low here, but it is strongly related to the score on its own (median score falls from about 8 for Low to about 5 for Very High stress). Several features overlap heavily (usage and unlocks correlate at 0.96), and permutation importance splits credit between correlated features, so low importance does not mean a feature is unrelated to the score.

## Limitations

- **Observational data**: the results show associations, not causes.
- **Overfitting**: train R² is about 0.99 against 0.90 on test. The test score is consistent with cross-validation, so the model generalizes on this data, but stronger regularization is a next step.
- **Data realism**: the dataset is very clean (no missing values, strong regular patterns, scores clustered at round values). It may be synthetic or heavily curated, so performance on real survey data could be lower.
- **Correlated features**: usage, unlocks, study hours and sleep overlap, which makes individual importances hard to separate.
- **Deployment note**: the country grouping happens outside the saved pipeline. Any new input must map countries outside the training top 10 to `Other` before prediction.

## Run it

```bash
pip install numpy pandas scikit-learn xgboost matplotlib seaborn joblib
jupyter notebook Student_MentalHealth_Regression.ipynb
```

Place the CSV in the same folder as the notebook. The tuned pipeline is saved as `mental_health_pipeline.joblib`.

## Next steps

- Residual and predicted-vs-actual analysis
- Stronger regularization to reduce the train/test gap
- Serve the model with a FastAPI `/predict` endpoint
- Validate on an independent dataset
