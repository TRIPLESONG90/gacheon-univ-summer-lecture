import os
import zipfile

with zipfile.ZipFile(os.path.join("./", 'dataset.zip'), 'r') as f:
    f.extractall()