# Auto Analytics: Advanced Estimation & Deployment 🛠️

![car_price_prediction](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F14886202%2F40cc46b822bdbeec638a90743892cdd3%2Findir%20(3).jpeg?generation=1704037226194977&alt=media)

This project focuses on using machine learning algorithms to estimate car prices. Various regression algorithms were implemented, including:

- Linear Regression
- Lasso Regression
- Ridge Regression
- Decision Tree
- Random Forest
- XGBoost

Model evaluation, grid-search, and cross-validation were performed, resulting in the following scores:

| Model             | R2    | MAE    | RMSE   | MAPE   |
|-------------------|-------|--------|--------|--------|
| XGBoost           | 0.921 | 2123.94| 3373.07| 0.132  |
| Random Forest     | 0.921 | 2252.57| 3374.97| 0.150  |
| Lasso             | 0.831 | 2818.00| 4954.25| 0.192  |
| Linear Regression | 0.830 | 2818.65| 4957.25| 0.192  |
| ElasticNet        | 0.830 | 2817.18| 4959.12| 0.192  |
| Decision Tree     | 0.816 | 3467.44| 5157.75| 0.221  |

The final models chosen were Random Forest and XGBoost. Feature importance was determined separately for each model to reduce feature counts. The models were saved using Pickle and converted into a Streamlit file for deployment outside the notebook environment. The Streamlit file was published on both AWS EC2 instances and the Streamlit website, enabling users to make predictions interactively.

## Access Links
- [Streamlit Live](https://auto-price-deployment.streamlit.app/)
- [AWS EC2](http://54.227.111.162:8502/)
- [GitHub Notebook Link](https://github.com/huseyincenik/auto_analytics_advanced_estimation_and_deployment)
- [Kaggle Notebook Link](https://www.kaggle.com/huseyincenik/auto-analytics-advanced-estimation-deployment)
- [LinkedIn](https://www.linkedin.com/in/huseyincenik/)
