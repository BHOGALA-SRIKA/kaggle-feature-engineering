import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

# 1. Setup & Data Loading
def load_data():
    df = pd.read_csv("ames.csv")
    X = df.copy()
    y = X.pop("SalePrice")
    return X, y

def score_dataset(X, y, model=XGBRegressor()):
    for colname in X.select_dtypes(["category", "object"]):
        X[colname], _ = X[colname].factorize()
    score = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_log_error")
    return np.sqrt(-1 * score.mean())

X, y = load_data()

# 2. Mutual Information (Ex 2)
def make_mi_scores(X, y):
    X_mi = X.copy()
    for colname in X_mi.select_dtypes(["object", "category"]):
        X_mi[colname], _ = X_mi[colname].factorize()
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X_mi.dtypes]
    mi_scores = mutual_info_regression(X_mi, y, discrete_features=discrete_features, random_state=0)
    return pd.Series(mi_scores, name="MI Scores", index=X_mi.columns).sort_values(ascending=False)

mi_scores = make_mi_scores(X, y) #

# 3. Mathematical Transforms (Ex 3)
X["LivLotRatio"] = X.GrLivArea / X.LotArea
X["Spaciousness"] = (X.FirstFlrSF + X.SecondFlrSF) / X.TotRmsAbvGrd
# Group transforms
X["MedNhbdArea"] = X.groupby("Neighborhood")["GrLivArea"].transform("median") #

# 4. Clustering with K-Means (Ex 4)
cluster_features = ["LotArea", "TotalBsmtSF", "FirstFlrSF", "SecondFlrSF", "GrLivArea"]
X_scaled = X.loc[:, cluster_features]
X_scaled = (X_scaled - X_scaled.mean(axis=0)) / X_scaled.std(axis=0)
kmeans = KMeans(n_clusters=10, n_init=10, random_state=0)
X["Cluster"] = kmeans.fit_predict(X_scaled) #

# 5. Principal Component Analysis (Ex 5)
pca_features = ["GarageArea", "YearBuilt", "TotalBsmtSF", "GrLivArea"]
X_pca_part = X.loc[:, pca_features]
X_pca_scaled = (X_pca_part - X_pca_part.mean(axis=0)) / X_pca_part.std(axis=0)
pca = PCA()
X_pca = pca.fit_transform(X_pca_scaled)
component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
X = X.join(pd.DataFrame(X_pca, columns=component_names, index=X.index)) #

# 6. Target Encoding (Ex 6)
# Smoothing helps prevent overfitting in target encoding
X["Neighborhood_Encoded"] = X.groupby("Neighborhood")["YearBuilt"].transform("mean") #

print(f"Final Model RMSLE: {score_dataset(X, y):.5f}")