
# Analysis 1: Data Summarization

## Objective
To obtain an overview of the dataset before in-depth analysis, summarizing the data’s main features. This step reveals basic distribution characteristics, including central tendency and variability for numerical variables, as well as frequency distributions for categorical variables. These summaries will provide foundational insights for data preprocessing and variable selection.

---

### 1. Descriptive Statistics for Quantitative Variables

**Content**:  
- **Descriptive statistics** summarize the primary features of numerical variables, providing key statistics. These statistics help understand central tendency, variability, and range. Specifically, this includes:
  - **Mean**: Shows the average level of the variable across observations.
  - **Standard Deviation**: Indicates the spread or variability of the variable; a higher standard deviation suggests a wider data distribution.
  - **Median**: Represents the central value in sorted data; useful for identifying data center, especially in skewed data.
  - **Minimum and Maximum**: Define the variable’s range, allowing for quick identification of extreme or outlier values.
  - **Quartiles (Q1, Q3)**: The first (Q1) and third (Q3) quartiles represent the 25th and 75th percentiles. The interquartile range (IQR = Q3 - Q1) can reveal the shape of the distribution and potential outliers.

**Execution Purpose**:  
Descriptive statistics offer a quantitative view of distribution, revealing patterns in central tendency through the mean and median, as well as variability through standard deviation and quartiles. The minimum and maximum values can help identify outliers, while quartiles provide insights into skewness. Overall, descriptive statistics support data cleaning, transformation, and analytical method selection.

---

### 2. Frequency Counts for Categorical Variables

**Content**:  
- **Frequency counts** analyze categorical variable distributions, tallying occurrences of each category. This step reveals distribution across categories and highlights potential imbalances. Specifically:
  - Count occurrences for each category, showing proportional differences between categories.
  - Identify any imbalances in category counts that might introduce biases in future analyses.

**Execution Purpose**:  
Frequency counts reveal the distribution of categorical data, helping identify unbalanced categories. If some categories are significantly underrepresented, it may be necessary to consider balancing methods, such as resampling or category merging, to improve the representativeness in further analysis.

---

### Significance of Data Summarization and Next Steps

- **Significance**: Descriptive statistics provide initial insights into the distribution, guiding data cleaning and transformation decisions. Frequency counts for categorical variables allow for identifying potential category imbalance issues, which is useful for planning resampling or balance methods.
- **Next Steps**: With data summarization completed, we’ll consider whether data transformation, handling of outliers, or balancing of categories is needed before further analysis.

```python
import pandas as pd

# Load the dataset
data = pd.read_csv("CSCS_data_anon.csv")


# 1. Descriptive Statistics for Quantitative Variables
quantitative_summary = data.describe().T  # Transpose for clearer viewing

# Display the descriptive statistics for quantitative variables
print("Quantitative Variable Summary:")
print(quantitative_summary)

# 2. Frequency Counts for Categorical Variables
categorical_summary = {}
for var in data.select_dtypes(include=['object', 'category']).columns:
    # Convert values to string to handle mixed types
    categorical_summary[var] = data[var].astype(str).value_counts()

# Convert the categorical summary to a DataFrame for easier viewing
categorical_summary_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in categorical_summary.items()]))

# Display the frequency counts for categorical variables
print("\nCategorical Variable Summary:")
print(categorical_summary_df)

```

#  Analysis 2: Data Visualization

## Objective
Use visualization to understand the data’s distribution, detect potential outliers, identify category imbalances, and observe relationships between variables. Visualization provides an intuitive view of the data structure, laying the groundwork for data preprocessing and model selection.

### 1. Distribution of Quantitative Variables

- **Content**: Plot each quantitative variable’s histogram and density curve to reveal the shape of the distribution.
  - **Histogram**: Displays the frequency distribution of each quantitative variable, helping identify central tendency, variability, and distribution symmetry.
  - **Density Curve**: Adds a smooth line over the histogram to show the probability density, revealing whether the variable approximates a normal distribution or has skewness or multimodal characteristics.

### 2. Frequency Distribution of Categorical Variables

- **Content**: Create bar charts for each categorical variable to display category counts.
  - **Bar Chart**: Shows the frequency of each category, making it easy to detect category imbalances. If some categories are significantly overrepresented, this may affect future analysis and may need balancing in later stages.

### 3. Pairwise Relationships (Inter-Variable Relationships)

- **Content**: Use scatter plot matrices or pairwise scatter plots to examine relationships between quantitative variables.
  - **Scatter Plot Matrix**: Shows pairwise relationships between variables, revealing any strong correlations or trends that may guide variable selection for regression.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("CSCS_data_anon.csv")

# 1. Distribution of Quantitative Variables
quantitative_vars = data.select_dtypes(include=['float64', 'int64']).columns

# Plot histograms and density curves
plt.figure(figsize=(15, 10))
for i, var in enumerate(quantitative_vars[:9], 1):  # Limit variable count to avoid overcrowding
    plt.subplot(3, 3, i)
    sns.histplot(data[var].dropna(), kde=True)  # Exclude missing values to avoid plot errors
    plt.title(f'Distribution of {var}')
    plt.xlabel(var)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 2. Frequency Distribution of Categorical Variables
categorical_vars = data.select_dtypes(include=['object', 'category']).columns

plt.figure(figsize=(15, 5))
for i, var in enumerate(categorical_vars[:3], 1):  # Limit variable count to avoid overcrowding
    plt.subplot(1, len(categorical_vars[:3]), i)
    data[var].value_counts().plot(kind='bar')
    plt.title(f'Frequency of {var}')
    plt.xlabel(var)
    plt.ylabel('Count')
plt.tight_layout()
plt.show()

# 3. Pairwise Relationships (Scatter Plot Matrix)
sns.pairplot(data[quantitative_vars[:5]].dropna())  # Select first 5 variables and exclude missing values
plt.show()

```

# Analysis 3: Simple Linear Regression

## Objective
Establish a simple linear regression model to quantify the linear impact of one independent variable on one dependent variable. This analysis helps determine if there is a significant linear relationship between two variables and measures the strength and direction of this relationship.

### 1. Variable Selection

- **Content**: Choose one numerical variable as the independent variable (X) and one as the dependent variable (Y).
- **Method**: Select variables based on prior knowledge or initial correlation analysis. In cases without specific requirements, two variables can be randomly selected.
- **Significance**: Choosing relevant variables affects the regression model’s effectiveness, as meaningful combinations increase explanatory power.

### 2. Data Preparation

- **Content**: Check for and handle missing values in the chosen variables to ensure data completeness.
- **Method**: Remove rows with missing values, or apply necessary transformations to enhance model stability.
- **Significance**: Ensuring data integrity improves accuracy in regression results and avoids bias.

### 3. Model Construction

- **Content**: Fit a simple linear regression model to represent the linear relationship between the independent and dependent variables.
- **Method**: Calculate the model’s slope and intercept, giving the regression equation: \( Y = b_0 + b_1X \), where \( b_0 \) is the intercept and \( b_1 \) is the slope.
- **Significance**: The slope, \( b_1 \), indicates the expected change in Y for each unit increase in X, while the intercept \( b_0 \) represents Y when X equals zero.

### 4. Model Evaluation

- **Content**: Evaluate the model fit using R² and Mean Squared Error (MSE).
- **Method**: Compute R² and MSE, where a high R² suggests strong explanatory power, and a low MSE indicates minimal prediction error.
- **Significance**: These evaluation metrics help assess model quality, with R² indicating how well X explains Y, and MSE showing prediction accuracy.

### 5. Residual Analysis

- **Content**: Analyze residuals to check if the model assumptions hold.
- **Method**: Plot residuals to see if they are normally distributed and lack a pattern.
- **Significance**: Residual analysis checks if assumptions like normality and homoscedasticity are met; if not, adjustments to the model or data may be necessary.

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("CSCS_data_anon.csv")
# 1. Calculate the correlation matrix and find highly correlated variable pairs
quantitative_vars = data.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = data[quantitative_vars].corr().abs()  # Absolute value for easy sorting
high_corr_pairs = correlation_matrix.unstack().sort_values(ascending=False)

# Filter out self-correlations and duplicate pairs
high_corr_pairs = high_corr_pairs[high_corr_pairs < 1].drop_duplicates()

# Select the highest correlated pair
x_var, y_var = high_corr_pairs.idxmax()

# 2. Data Preparation (Remove rows with missing values)
filtered_data = data[[x_var, y_var]].dropna()

# 3. Build the Linear Regression Model
X = filtered_data[[x_var]]
Y = filtered_data[y_var]
model = LinearRegression()
model.fit(X, Y)

# 4. Model Evaluation
slope = model.coef_[0]
intercept = model.intercept_
predictions = model.predict(X)
r_squared = r2_score(Y, predictions)
mse = mean_squared_error(Y, predictions)

# Output results
print(f"Selected Variables with High Correlation: X = {x_var}, Y = {y_var}")
print(f"Regression Equation: Y = {intercept:.2f} + {slope:.2f} * X")
print(f"R-squared: {r_squared:.4f}")
print(f"Mean Squared Error: {mse:.2f}")

# 5. Residual Analysis
residuals = Y - predictions
plt.figure(figsize=(10, 5))
plt.scatter(predictions, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title("Residual Plot")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.show()

```

```python

```
