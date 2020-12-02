from joblib import load
import pandas as pd
import numpy as np
import argparse
import json


# load the trained model
model = load('./model/clf_rf_20201124')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=argparse.FileType('r', encoding='UTF-8'), required=True)
    args = parser.parse_args()
    
    x = pd.read_csv(args.infile, header=None)
    
    output = model.predict(x)
    
    print({'status': 200,
           'response': output})
    
    
    
