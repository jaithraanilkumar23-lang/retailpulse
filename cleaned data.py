import pandas as pd

df = pd.read_csv("retail_customer_segmentation.csv")

print(df.head())
import pandas as pd

# Load dataset
df = pd.read_csv("retail_customer_segmentation.csv")

# First 5 rows
print(df.head())

# Dataset information
print(df.info())

# Missing values
print(df.isnull().sum())

# Statistical summary
print(df.describe())
import pandas as pd

# Load dataset
df = pd.read_csv("retail_customer_segmentation.csv")

# Check missing values
print(df.isnull().sum())

# Fill missing numerical values with mean
df.fillna(df.mean(numeric_only=True), inplace=True)

# Check again
print("\nAfter Cleaning:\n")
print(df.isnull().sum())

# Save cleaned dataset
df.to_csv("cleaned_retail_customer_segmentation.csv", index=False)

print("\nData cleaning completed!")