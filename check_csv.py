import pandas as pd

DATA_PATH = "data/pages.csv.gz"

df = pd.read_csv(
    DATA_PATH,
    compression="gzip",
    nrows=5,
    encoding="utf-8",
    low_memory=False,
)

print("Первые строки:")
print(df.head())

print()
print("Колонки:")
print(df.columns.tolist())

print()
print("Типы данных:")
print(df.dtypes)
