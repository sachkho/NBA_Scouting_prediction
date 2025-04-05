# NBA Scouting Algorithm for Drafting College Players

## Introduction

There are multiple ways to approach the problem of NBA scouting. An NBA team with a high first-round pick can safely choose a top-20 national prospect, usually ensuring a solid player for the next decade. However, the goal here is to challenge national rankings with a custom scouting method that considers factors overlooked by traditional approaches.

Scouting today often relies on flashy statistics that don't necessarily translate into professional success. Many top draft picks ("draft busts") average impressive college numbers but fail to transition successfully to the NBA. This project attempts to build an algorithm that focuses on long-term performance indicators, aiming to find undervalued prospects.

## The Dataset

The first step was finding data containing both player statistics and their team's performance during their college careers. After struggling to find structured NCAA datasets on platforms like Kaggle and HuggingFace, the dataset used in this project was sourced from [this GitHub repository](https://github.com/RobertoCurti/basketballData/).

The dataset is a multi-table format with advanced player statistics and team records.

![Dataset Architecture](architecture_dataset.png)

## Building Our Dataset

To predict how a college player will perform in the NBA, we use John Hollinger's Player Efficiency Rating (PER) as our target metric:

```
PER = (uPER * League Pace Factor / League Average uPER) * 15
```

More details are available on [Basketball Reference](https://www.basketball-reference.com/about/per.html).

We had to manually merge multiple tables to match players with their NBA and college statistics and compute average college PERs. A key challenge was ensuring correct player ID mappings across different years, as players often had multiple IDs.

### Final Dataset Example

| Player       | Player.id | stat.PERColl | stat.PERNBA | ... | Height, Weight, etc. |
|--------------|-----------|--------------|-------------|-----|-----------------------|
| NBA Player 1 | 1408      | 14.3         | 21.0        | ... | ...                   |
| NBA Player 2 | 3027      | 12.6         | 13.1        | ... | ...                   |

## Dealing with Missing Values

To handle missing data, we used a K-Nearest Neighbors (KNN) imputer, which estimates missing values based on similar players. Players missing over 40% of data were dropped, and the remaining features were standardized with `StandardScaler()`.

After cleaning, the dataset included **902 players** with **42 features**.

## Data Visualization

We used a pairplot to visualize feature correlations with the target variable (success, defined as NBA PER > 0.45):

```python
seaborn.pairplot(dataset, hue="success")
```

![Pairplot](pairplot%20blue%20and%20grey.png)

## Choosing the Model

Since we're predicting a numeric value (PER), this is a **regression** task. We selected **XGBoost**, a tree-based ensemble model known for strong performance on small datasets and its ability to provide feature importance metrics.

## Results

Initial XGBoost model results:

- **RMSE**: 1.09
- **R²**: 0.01
- **MAE**: 0.77

These results are not satisfying. R² = 0.01 means the model explains virtually none of the variance in PER.

![Prediction Plot](predicted%20PER%20vs%20actual%20PER.png)

Potential issues:
- Small dataset size
- Noisy or irrelevant features
- Heavy imputation

### Feature Importance

Some irrelevant features (e.g., `historical.id_ncaa`) were incorrectly given high importance by the model:

![Feature Importance](feature%20importance.png)

Removing these features or applying **dimensionality reduction** (like PCA) might improve results.

## European Player Trend

In recent years, European players have dominated the NBA (e.g., Giannis, Jokic, Doncic). These players often come from leagues with slower pace and more structured systems, mirroring 80s/90s NBA styles. Scouting should therefore expand to include European prospects and consider league context.

## Conclusion

Predicting NBA success from college statistics is complex and depends heavily on the quality and completeness of the dataset. While XGBoost performed poorly here, improving the feature set, expanding the dataset (including European leagues), or exploring more advanced models (e.g., TabPFN, neural nets) could yield better results.

In practice, feature engineering and understanding the context behind each player's stats will always remain crucial components of a successful scouting algorithm.

---

*Project by Sacha Khosrowshahi*