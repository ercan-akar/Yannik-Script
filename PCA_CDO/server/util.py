import numpy as np
import pandas as pd

def make_serializable(d):
    if isinstance(d, list):
        out = []
        for v in d:
            if isinstance(v, dict) or isinstance(v, list):
                out.append(make_serializable(v))
            elif isinstance(v, np.ndarray):
                out.append(v.tolist())
            elif isinstance(v, pd.DataFrame):
                out.append(v.to_dict())
            else:
                out.append(v)
    
    elif isinstance(d, dict):
        out = {}
        for k, v in d.items():
            if isinstance(v, dict) or isinstance(v, list):
                out[k] = make_serializable(v)
            elif isinstance(v, np.ndarray):
                out[k] = v.tolist()
            elif isinstance(v, pd.DataFrame):
                out[k] = v.to_dict()
            else:
                out[k] = v

    return out