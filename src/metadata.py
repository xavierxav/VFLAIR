import pandas as pd
import numpy as np
from pymfe.mfe import MFE
from sklearn.tree import DecisionTreeClassifier

# Load your dataset
data = pd.read_csv(r'.\load\share_dataset\tabledata\UCI_Credit_Card.csv')

# Extract metafeatures using PyMFE
mfe = MFE(groups=["general", "statistical", "info-theory", "landmarking", "model-based"])
mfe.fit(data.iloc[:, 1:-1].values, data.iloc[:, -1].values)
metafeatures = mfe.extract()

# Sample metafeatures (for illustration, use actual metafeatures in practice)
sample_metafeatures = np.array(metafeatures)[1]

# Decision tree model based on the paper's decision tree
def choose_model(metafeatures):
    if metafeatures[0] <= 4211:
        if metafeatures[0] <= 1132:
            if metafeatures[1] <= 65:
                if metafeatures[2] <= 0.044:
                    return "XGBoost"
                else:
                    return "CatBoost"
            else:
                if metafeatures[3] <= 0.025:
                    return "SAINT"
                else:
                    return "ResNet"
        else:
            if metafeatures[4] <= 0.792:
                return "TabPFN"
            else:
                return "ResNet"
    else:
        if metafeatures[5] <= 0.003:
            return "XGBoost"
        else:
            return "CatBoost"

# Predict the best model for the dataset
best_model = choose_model(sample_metafeatures)
print(f"The best model for your dataset is: {best_model}")
