import pandas as pd

# 1. Generate example data
data = {
    "Color": ["Red", "Blue", "Green", "Red", "Blue"],
    "Size": ["S", "M", "L", "M", "S"],
    "Price": [100, 150, 200, 100, 150],
}
df = pd.DataFrame(data)

print("--- Original DataFrame ---")
print(df)
print("-" * 30)

# 2. Apply One-Hot Encoding to the 'Color' and 'Size' columns
# Using drop_first=True removes the first category to prevent multicollinearity.
df_encoded = pd.get_dummies(df, columns=["Color", "Size"], drop_first=True)

print("--- DataFrame After One-Hot Encoding ---")
print(df_encoded)
