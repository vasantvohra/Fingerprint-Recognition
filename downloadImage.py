import pandas as pd
import json
import os
df = pd.read_csv('shuraimage.csv',header = None)
new_header = df.iloc[0]
df=df[1:]
df.columns = new_header

def download(Id=9708,finger ="left_thumb"):
     filename = df[df.id==str(Id)]
     file = filename[finger]
     try:
         for i in file:
             print(i)
             os.system(f"curl -o {i} https://magostech.net/unproject/uploads/{i}")
             print("Download successful")
             return i
     except:
         print("file not found")
