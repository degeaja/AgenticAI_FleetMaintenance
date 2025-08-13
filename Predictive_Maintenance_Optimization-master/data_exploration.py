import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("fleet_train.csv")

print(df.head)

# Show shape and info
print(f"Shape: {df.shape}")
print("\nColumn Types:\n", df.dtypes)

# Check nulls
print("\nMissing Values:\n", df.isnull().sum())

# Basic statistics
print("\nDescriptive Stats:\n", df.describe())

# Unique values per column
print("\nUnique Values per Column:")
for col in df.columns:
    print(f"{col}: {df[col].nunique()}")

# Target distribution
print("\nTarget Value Counts (Maintenance_flag):")
print(df["Maintenance_flag"].value_counts(normalize=True))

# Plot target distribution
sns.countplot(data=df, x="Maintenance_flag")
plt.title("Maintenance Flag Distribution")
plt.show()
