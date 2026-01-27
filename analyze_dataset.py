import pandas as pd

# Load the dataset
df = pd.read_csv('01_Bala-kanda-output.txt', sep='\t', header=None, 
                 names=['entity', 'kanda', 'chapter', 'text'])

print(f'Total rows: {len(df)}')
print(f'\nUnique chapters: {df["chapter"].nunique()}')
print(f'\nChapter distribution:')
print(df['chapter'].value_counts())
print(f'\nSample data:')
print(df.head())
