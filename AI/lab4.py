import pandas as pd
df = pd.read_csv('path.csv')
df.head()
#upload file
# Show first and last 5 rows
print("First 5 rows:\n", df.head())
print("\nLast 5 rows:\n", df.tail())
# Print all the Iris-setosa details
if 'species' in df.columns:
    print("\nAll Iris-setosa details:")
    print(df[df['species'] == 'Iris-setosa'])
# Count total species per classification
    print("\nSpecies count:")
    print(df['species'].value_counts())
# Sort all species by Sepal Length column
if 'sepal_length' in df.columns:
    print("\nSorted by Sepal Length:")
    print(df.sort_values(by='sepal_length'))
# How many species have sepal width > 4
if 'sepal_width' in df.columns:
    print("\nNumber of species with sepal width > 4:")
    print(len(df[df['sepal_width'] > 4]))
# Remove row number 150 (index 149)
if 149 in df.index:
    df.drop(index=149, inplace=True)
    print("\nRow number 150 (index 149) removed.")
else:
    print("\nIndex 149 does not exist in the dataset.")
# Select only first 9 columns
print("\nFirst 9 columns (or all if less):")
print(df.iloc[:, :9])
# Select all columns except column 3 (index 2)
if df.shape[1] > 2:
    print("\nAll columns except column 3:")
    print(df.drop(df.columns[2], axis=1))
else:
    print("\nDataset has less than 3 columns, skipping drop column.")
