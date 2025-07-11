import sys
import os
path_to_src = "/Users/paoloriotino/Documents/GitHub/NLP_prediction-and-forecasting/chronos-forecasting/src"
sys.path.append(os.path.abspath(path_to_src))

import pandas as pd
import torch
from chronos.chronos import ChronosPipeline

pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cpu",
    torch_dtype=torch.bfloat16,
)

df = pd.read_csv("https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv")

# context must be either a 1D tensor, a list of 1D tensors,
# or a left-padded 2D tensor with batch as the first dimension
context = torch.tensor(df["#Passengers"])
embeddings, tokenizer_state = pipeline.embed(context)