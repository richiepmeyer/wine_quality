# Taste of Fine Wine


# Project Goals:
- Determine drivers of quality
- Utilize drivers to create a model to predict quality
- Develop deeper understanding on how drivers influence quality


# Project description:
This project undertakes the task of uncovering what characteristics influence a wine's quality. We are looking at Vinho Verde wines that originate from Minho, a northern province in Portugal. Vinho Verde is known for their value stance in making great wines.


# Initial Hypothesis
Our initial hypthosis is that the proportion of certain characterics influence the quality of wine. Most notably, alcohol and sugar, total acidity, and chlorides. 


# Questions
1. The proportion of acid to (alcohol + sugar) affects quality
2. White and red wines' quality is different
3. The proportion of acid to chlorides affects quality
4. Density affects the quality of wine 

# The Plan

Project planning (lay out your process through the data science pipeline)
1. Acquire data, clean data
2. Explore and deal with outliers
3. Split data
4. Explore data on unscaled train set 
    -train_scaled , train_unscaled
    -Cluster on scaled data
        -density, volatile acidity, and alcohol
        -add clusters to X_train, X_val, and X_test
5. Create questions and visuals and statistical tests
6. Model using Regression
    -Feature selection (recurvsive, etc.)
7. Create final notebook



# Data dictionary

| Feature | Definition |
| :-- | :-- |
| fixed_acidity | predominant fixed acids found in wines are tartaric, malic, citric, and succinic |
| volatile_acidity | the steam distillable acids present in wine, primarily acetic acid but also lactic, formic, butyric, and propionic acids |
| citric_acid | citric acid |
| residual_sugar | natural sugars left over after fermentation | 
| chlorides | compound that plays a major role in a wine's salt profile |
| free_sulfur_dioxide | sulphur dioxide ions that are not chemically bound to other chemicals |
| total_sulfur_dioxide | free sulfur dioxide plus those that are not bound to other chemicals |
| pH | scale used to measure acidity |
| sulphates | salts of sulfuric acid |
| alcohol | alcohol |
| quality | rating of wine |
| color_red | 0 = white wine<br>1 = red wine<br>|
| acid_alc_sugar | Total acidity divided by alcohol plus sugar |
| acid_chlor |Total acidity divided by chlorides |


# Steps to Reproduce
1. Clone this repo.
2. To acquire data, go to data.world and download red wine and white wine data.
    - link: https://data.world/food/wine-quality
3. Use functions in acquire.py to upload data
5. Use functions in explore.py to clean and explore data.
5. Use functions in visual.py to plot the charts and stats test.
5. Use same configurations for models.
6. Use functions in model.py to evaluate models.


# Conclusion

- Determined that alcohol, volatile acidity, density, and chlorides have significant influence on a wine's quality.
- Our model improves upon the baseline by 0.13 on the RMSE value.

# Recommendations

Use different features such as sulfur dioxide and pH to see how they influence quality.
Gather data on tannins to see how they influence quality.

