import pandas as pd
from datasets import load_dataset

dataset = load_dataset("davidberenstein1957/healthcare-consults")
df = pd.DataFrame(dataset['train'])
df.to_json("../data/healthcare-consults.json", orient="records", force_ascii=False, indent=2)