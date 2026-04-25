# Business Case Analysis — Promotion Effectiveness at a Fashion Retail Chain

## B1. Problem Formulation

### B1(a) — ML Problem Formulation

The target variable is items_sold — the number of items sold at a store in a
given month under a specific promotion. The candidate input features include
store attributes such as store size, location type, and monthly footfall;
promotion type (Flat Discount, BOGO, Free Gift, Category-Specific Offer,
Loyalty Points Bonus); contextual features such as local competition density
and customer demographics; and temporal features such as month and season.

This is a supervised regression problem because the target variable is
continuous and numerical. The goal is to predict how many items will be sold
under each promotion scenario, and then select the promotion that maximises
the predicted sales volume for each store each month.

### B1(b) — Why Items Sold is a Better Target Variable

Using total sales revenue as the target variable is unreliable because revenue
is influenced by price fluctuations, discounts, and promotional offers
themselves. For example, a Flat Discount promotion may generate lower revenue
per item but drive higher sales volume. If the model targets revenue, it may
incorrectly favour high-price low-volume promotions over high-volume ones.

Items sold is a more stable and reliable target because it directly measures
customer response to a promotion without being distorted by pricing effects.
This illustrates the broader principle that target variables should measure
the actual business outcome of interest as directly and cleanly as possible,
free from confounding factors that are themselves influenced by the input
features.

### B1(c) — Alternative Modelling Strategy

A single global model across all 50 stores would ignore the fact that stores
in different locations respond very differently to the same promotion. A better
strategy would be to train separate models for each store segment — for example
grouping stores by location type (urban, semi-urban, rural) and training one
model per group. This is known as a segmented or hierarchical modelling
approach. Alternatively, store-level fixed effects or embeddings could be
included as features in a single model to allow it to learn store-specific
behaviour while still sharing information across stores.


-------------------------------------------------------------------------------------------------

## B2. Data and EDA Strategy

### B2(a) — Joining Tables and Dataset Grain

The four tables would be joined as follows. The transactions table is the base
table containing one row per transaction. Store attributes are joined to
transactions using store_id as the key. Promotion details are joined using
promotion_type or a promotion_id key. The calendar table is joined using the
transaction date to bring in weekend and festival flags.

The grain of the final modelling dataset would be one row per store per month
per promotion type. Before modelling, transactions would be aggregated to
compute total items sold, total revenue, and average basket size for each
store-month-promotion combination. This aggregation ensures the model learns
at the decision-making level, which is monthly promotion assignment per store.

### B2(b) — EDA Strategy

Four key analyses would be performed before modelling. First, a bar chart of
average items sold by promotion type would reveal which promotions perform
best overall and whether any promotion consistently underperforms. This would
influence whether promotion type should be treated as a simple categorical
feature or whether interaction terms with location are needed.

Second, a heatmap of average items sold by promotion type and location type
would identify whether certain promotions work better in urban versus rural
stores. If strong interactions exist, separate models per location type would
be justified.

Third, a time series plot of monthly items sold across all stores would reveal
seasonality trends and festival effects. This would inform the creation of
temporal features such as month, quarter, and festival flags.

Fourth, a box plot of items sold by store size would show whether store size
significantly affects sales volume, helping decide whether store size should
be included as a feature or used to segment the data.

### B2(c) — Handling Promotion Imbalance

If 80% of transactions occurred without any promotion, the model will be
heavily biased towards predicting outcomes for the no-promotion case and may
poorly estimate the effect of specific promotions. To address this, the
training data could be resampled to balance promoted and non-promoted
transactions using oversampling techniques. Alternatively, the model could be
trained only on promoted transactions if the goal is specifically to compare
promotion types. Sample weights could also be applied to give promoted
transactions greater influence during training.


-----------------------------------------------------------------------------------------------

## B3. Model Evaluation and Deployment

### B3(a) — Train-Test Split and Evaluation Metrics

With three years of monthly store-level data across 50 stores, the train-test
split should be temporal. The first two years of data would be used for
training and the most recent year for testing. A random split is inappropriate
because it would allow the model to train on future months and test on past
months, causing data leakage and producing overly optimistic performance
estimates.

The evaluation metrics would be RMSE and MAE. RMSE penalises large prediction
errors more heavily, which is important because significantly over or
under-stocking a store due to a wrong promotion recommendation has serious
operational consequences. MAE gives an interpretable average error in the same
units as items sold, making it easy to communicate to the marketing team how
far off the predictions typically are.

### B3(b) — Feature Importance for Explaining Recommendations

To investigate why the model recommends Loyalty Points Bonus for Store 12 in
December but Flat Discount in March, feature importances from the Random Forest
model would be examined. If is_festival and month are among the top features,
this would explain the December recommendation — December likely coincides with
a festive period where loyalty rewards drive higher engagement. In March, the
absence of festivals and different customer behaviour may make immediate
discounts more effective. These findings would be communicated to the marketing
team using a simple chart showing the top features driving each recommendation,
with plain English explanations linking feature values to the promotion choice.

### B3(c) — End-to-End Deployment Process

After training, the model would be saved using Python's joblib library so it
can be loaded without retraining. At the start of every month, new store and
calendar data for the upcoming month would be prepared in the same format as
the training data, including all engineered features. This data would be fed
into the saved model pipeline to generate promotion recommendations for all
50 stores, which would then be shared with the marketing team via a dashboard
or automated report.

To monitor model performance, actual items sold would be recorded each month
and compared against the model's predictions. If RMSE or MAE consistently
exceeds an acceptable threshold, or if the distribution of input features
shifts significantly compared to the training data, this would trigger a
retraining process using the most recent available data.