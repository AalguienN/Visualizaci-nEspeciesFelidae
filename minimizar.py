import pandas as pd

# columnas que realmente usas en tu app
cols = [
  "countryCode","scientificName","vernacularName",
  "decimalLatitude","decimalLongitude",
  "eventDate","year","month","habitat",
  "elevation","depth",
  "coordinateUncertaintyInMeters","individualCount","gbifID"
]
df = pd.read_csv("gatos_dataset.txt", sep="\t", usecols=cols, dtype=str)

# escribe en formato Parquet (mucho más comprimido) o CSV gzip
df.to_parquet("gatos_light.parquet", compression="snappy")
# ó
df.to_csv("gatos_light.csv", index=True, sep="\t")
