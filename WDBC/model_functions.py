from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.svm import SVC
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


"""
PREPROCESSING
"""

def preprocess_data(
    file='data/wdbc.data',
    corr_threshold=0.3,
    test_size=0.2,
    random_state=42
):
    # -------------------------------------
    # 1) Load dataset and rename columns
    # -------------------------------------
    df = pd.read_csv(file, header=None)
    cols = ["id", "diagnosis"] + [
        f"{feat}_{stat}"
        for stat in ["mean", "se", "worst"]
        for feat in ["radius", "texture", "perimeter", "area",
                     "smoothness", "compactness", "concavity",
                     "concave_points", "symmetry", "fractal_dimension"]
    ]
    df.columns = cols

    # -------------------------------------
    # 2) Numerical encoding for correlation filtering
    # -------------------------------------
    df["diagnosis"] = df["diagnosis"].map({"B": 0.0, "M": 1.0})

    correlated_columns = ["diagnosis"]
    for col in df.drop(columns=["diagnosis", "id"]).columns:
        r, _ = pointbiserialr(df["diagnosis"], df[col].astype(float))
        if r > corr_threshold:
            correlated_columns.append(col)

    df = df[correlated_columns]

    # Convert back to category labels
    df["diagnosis"] = df["diagnosis"].map({0.0: "Benign", 1.0: "Malignant"}).astype("category")

    # -------------------------------------
    # 3) Split into X and y
    # -------------------------------------
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # -------------------------------------
    # 4) Scaling
    # -------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # -------------------------------------
    # 5) PCA
    # -------------------------------------
    pca = PCA(n_components= 5)
    pca.fit(X_train_scaled)

    X_train_pca = pca.transform(X_train_scaled)
    X_test_pca  = pca.transform(X_test_scaled)

    # -------------------------------------
    # 6) Return everything useful
    # -------------------------------------
    return {
        "df": df,                            
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "X_train_pca": X_train_pca,
        "X_test_pca": X_test_pca
    }

"""
MODEL 1: LOGISTIC REGRESSION
"""

def logistic_cross_validation(X, y,
                              penalty='l2',
                              C=10,
                              solver='liblinear',
                              max_iter=1000,
                              k=30):
    """
    Manual Stratified K-fold cross-validation for Logistic Regression.

    Parameters:
        X, y : array or DataFrame
        penalty : 'l1', 'l2', etc.
        C : inverse of regularization strength
        solver : optimization algorithm ('lbfgs','liblinear', etc.)
        max_iter : number of iterations
        k : number of folds

    Returns:
        metrics (dict): CV means and stds
        preds (array): concatenated predictions from all folds
    """

    # Convert string labels to numeric if necessary
    if y.dtype == object or isinstance(y[0], str):
        y = np.where(y == "Malignant", 1, 0)

    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    accuracies, f1s, recalls, aucs = [], [], [], []
    all_preds = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"Fold {fold+1}/{k}")

        # ---- SPLIT (pandas OR numpy) ----
        if hasattr(X, "iloc"):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
        else:
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

        # ---- SCALING inside fold ----
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # ---- MODEL ----
        lr = LogisticRegression(
            penalty=penalty,
            C=C,
            solver=solver,
            max_iter=max_iter,
            # class_weight='balanced',   # <— optionally uncomment if you want fairness across classes
        )

        lr.fit(X_train, y_train)

        # ---- PREDICT ----
        y_pred = lr.predict(X_val)
        y_proba = lr.predict_proba(X_val)[:, 1]

        # ---- METRICS ----
        accuracies.append(accuracy_score(y_val, y_pred))
        f1s.append(f1_score(y_val, y_pred, pos_label=1))
        recalls.append(recall_score(y_val, y_pred, pos_label=1))
        aucs.append(roc_auc_score(y_val, y_proba))

        all_preds.extend(y_pred)

    metrics = {
        "accuracy_mean": np.mean(accuracies),
        "accuracy_std": np.std(accuracies),
        "f1_mean": np.mean(f1s),
        "f1_std": np.std(f1s),
        "recall_mean": np.mean(recalls),
        "recall_std": np.std(recalls),
        "auc_mean": np.mean(aucs),
        "auc_std": np.std(aucs),
        "recalls_per_fold": recalls
    }

    return metrics, np.array(all_preds)

"""
MODEL 2: SVM
"""

def svm_cross_validation(X, y,
                         kernel='rbf',
                         C=1,
                         gamma='scale',
                         probability=True,
                         degree=2,
                         k=30):

    if y.dtype == object or isinstance(y[0], str):
        y = np.where(y == "Malignant", 1, 0)

    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    accuracies, f1s, recalls, aucs = [], [], [], []
    all_preds = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"Fold {fold+1}/{k}")

        # Split (pandas vs numpy safe)
        if hasattr(X, "iloc"):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
        else:
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

        # Scaling inside fold
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # Train SVM
        svm = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, probability=probability)
        svm.fit(X_train, y_train)

        # Predict
        y_pred = svm.predict(X_val)
        y_proba = svm.predict_proba(X_val)[:, 1]

        # Metrics
        accuracies.append(accuracy_score(y_val, y_pred))
        f1s.append(f1_score(y_val, y_pred, pos_label=1))
        recalls.append(recall_score(y_val, y_pred, pos_label=1))
        aucs.append(roc_auc_score(y_val, y_proba))

        all_preds.extend(y_pred)

    metrics = {
        "accuracy_mean": np.mean(accuracies),
        "accuracy_std": np.std(accuracies),
        "f1_mean": np.mean(f1s),
        "f1_std": np.std(f1s),
        "recall_mean": np.mean(recalls),
        "recall_std": np.std(recalls),
        "auc_mean": np.mean(aucs),
        "auc_std": np.std(aucs),
        "recalls_per_fold": recalls
    }

    return metrics, np.array(all_preds)

"""
MODEL 3: MLP
"""

def mlp_cross_validation(X, y,
                         hidden_layer_sizes=(100,50),
                         activation='logistic',
                         solver='adam',
                         alpha=0.01,
                         learning_rate='constant',
                         learning_rate_init=0.1,
                         max_iter=2000,
                         k=30):
    """
    Manual Stratified K-fold cross-validation for an MLP classifier.
    """

    # Convert labels to numeric if string-based
    if y.dtype == object or isinstance(y[0], str):
        y = np.where(y == "Malignant", 1, 0)

    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    accuracies, f1s, recalls, aucs = [], [], [], []
    all_preds = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"Fold {fold+1}/{k}")

        # ---- SPLIT (pandas OR numpy) ----
        if hasattr(X, "iloc"):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
        else:
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

        # ---- SCALING (inside fold, avoid leakage) ----
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # ---- MODEL ----
        mlp = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter
        )

        mlp.fit(X_train, y_train)

        # ---- PREDICT ----
        y_pred = mlp.predict(X_val)
        y_proba = mlp.predict_proba(X_val)[:, 1]

        # ---- METRICS ----
        accuracies.append(accuracy_score(y_val, y_pred))
        f1s.append(f1_score(y_val, y_pred, pos_label=1))
        recalls.append(recall_score(y_val, y_pred, pos_label=1))
        aucs.append(roc_auc_score(y_val, y_proba))

        all_preds.extend(y_pred)

    metrics = {
        "accuracy_mean": np.mean(accuracies),
        "accuracy_std": np.std(accuracies),
        "f1_mean": np.mean(f1s),
        "f1_std": np.std(f1s),
        "recall_mean": np.mean(recalls),
        "recall_std": np.std(recalls),
        "auc_mean": np.mean(aucs),
        "auc_std": np.std(aucs),
        "recalls_per_fold": recalls
    }

    return metrics, np.array(all_preds)

"""
MODEL 4: HYBRID MODEL
"""

class BottleneckMLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes=(64,32), bottleneck_dim=8):
        super().__init__()

        layers = []
        prev_dim = input_dim
        
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        
        layers.append(nn.Linear(prev_dim, bottleneck_dim))

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(bottleneck_dim, 1)

    def forward(self, x):
        z = self.feature_extractor(x)
        out = torch.sigmoid(self.classifier(z))
        return out, z

def hybrid_nn_svm_cv(X, y,
                     bottleneck_dim=16,
                     epochs=100,
                     lr=0.001,
                     k=10):
    """
    Stratified K-fold cross validation for Hybrid NN + SVM.
    """

    # Convert labels to numpy and encode Malignant=1 if necessary
    if isinstance(y[0], str) or y.dtype == object:
        y = np.where(y == "Malignant", 1, 0)
    else:
        y = np.array(y).astype(int)

    # Metrics storage
    accuracies, f1s, recalls, aucs = [], [], [], []
    preds_all = []

    # Stratified K-fold
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"\nFold {fold+1}/{k}")

        # -------- SPLIT --------
        if hasattr(X, "iloc"):  # pandas DataFrame
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        else:  # numpy array
            X_train, X_val = X[train_idx], X[val_idx]

        y_train, y_val = y[train_idx], y[val_idx]

        # -------- SCALING (INSIDE FOLD TO AVOID LEAKAGE) --------
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # -------- TORCH DATA --------
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)

        # -------- DEFINE MODEL --------
        model = BottleneckMLP(input_dim=X_train.shape[1], bottleneck_dim=bottleneck_dim)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # -------- TRAIN NEURAL NETWORK --------
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs, _ = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

        # -------- EXTRACT LATENT FEATURES --------
        model.eval()
        with torch.no_grad():
            _, Z_train = model(X_train_tensor)
            _, Z_val = model(X_val_tensor)

        Z_train_np = Z_train.numpy()
        Z_val_np = Z_val.numpy()

        # -------- SVM CLASSIFIER --------
        svm = SVC(kernel='linear', C=10, gamma=1, probability=True)
        svm.fit(Z_train_np, y_train)

        # -------- PREDICTIONS --------
        y_pred = svm.predict(Z_val_np)
        y_proba = svm.predict_proba(Z_val_np)[:, 1]

        # -------- METRICS --------
        accuracies.append(accuracy_score(y_val, y_pred))
        f1s.append(f1_score(y_val, y_pred, pos_label=1))
        recalls.append(recall_score(y_val, y_pred, pos_label=1))
        aucs.append(roc_auc_score(y_val, y_proba))

        preds_all.extend(y_pred)

    # -------- RESULTS --------
    metrics_dict = {
        "accuracy_mean": np.mean(accuracies),
        "accuracy_std": np.std(accuracies),
        "f1_mean": np.mean(f1s),
        "f1_std": np.std(f1s),
        "recall_mean": np.mean(recalls),
        "recall_std": np.std(recalls),
        "auc_mean": np.mean(aucs),
        "auc_std": np.std(aucs),
        "recalls_per_fold": recalls
    }

    return metrics_dict, np.array(preds_all)
