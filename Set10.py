import pandas as pd
import numpy as np

# Step 1: Load the dataset
df = pd.read_csv("SET_10.csv")

# Step 2: Replace missing numerical values with column mean
df.fillna(df.mean(numeric_only=True), inplace=True)

# Step 3: Remove duplicate records
df.drop_duplicates(inplace=True)

# Step 4: Apply Min-Max normalization to numerical columns
numerical_cols = df.select_dtypes(include=np.number).columns
df[numerical_cols] = (df[numerical_cols] - df[numerical_cols].min()) / (df[numerical_cols].max() - df[numerical_cols].min())

# Step 5: Discretize Math scores into grades
def get_math_grade(score):
    if score < 0.5:
        return 'F'
    elif score < 0.6:
        return 'D'
    elif score < 0.7:
        return 'C'
    elif score < 0.8:
        return 'B'
    else:
        return 'A'

df['Math_Grade'] = df['Math'].apply(get_math_grade)

# Step 6: Smooth noisy Math data using binning (equal-width binning into 4 bins)
df['Math_Bin'] = pd.cut(df['Math'], bins=4, labels=False)
bin_means = df.groupby('Math_Bin')['Math'].mean()
df['Math_Smoothed'] = df['Math_Bin'].map(bin_means)

# Display the final processed DataFrame
print(df)

# Optional: Save the cleaned data to a new CSV
df.to_csv("SET_10_cleaned.csv", index=False)
