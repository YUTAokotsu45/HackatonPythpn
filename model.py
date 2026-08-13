import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import accuracy_score, r2_score, f1_score, precision_score, recall_score, mean_absolute_error, mean_squared_error


# --- METRIQUES (ajout) ---

def evaluate(ytrue, ypred, task):

    if task == "classification":

        return {
            "accuracy": float(accuracy_score(ytrue, ypred)),
            "precision_macro": float(precision_score(ytrue, ypred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(ytrue, ypred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(ytrue, ypred, average="macro", zero_division=0)),
        }

    return {
        "r2": float(r2_score(ytrue, ypred)),
        "mae": float(mean_absolute_error(ytrue, ypred)),
        "rmse": float(np.sqrt(mean_squared_error(ytrue, ypred))),
    }


# --- FONCTION COMMUNE (SmartData) ---

def SmartData(df : pd.DataFrame, model, x, y):

    x = x.copy()

    for column in x.columns:

        if pd.api.types.is_datetime64_any_dtype(x[column]):

            x[column] = x[column].astype("int64") // 10**9

    xtr, xtst, ytr, yts = train_test_split(
        x,
        y,
        test_size=0.3,
        random_state=42
    )

    model.fit(xtr, ytr)

    scaler = round(len(df.columns) / 3)

    importance = model.feature_importances_

    df = pd.DataFrame({
        "column": xtr.columns,
        "importance": importance
    })

    best_cols = df.sort_values(
        by="importance",
        ascending=False
    ).head(scaler)["column"].tolist()

    print("test de decision cologne : ", best_cols)

    xtr = xtr[best_cols]
    xtst = xtst[best_cols]

    model.fit(xtr, ytr)

    return model, xtr, xtst, ytr, yts, best_cols


# ============================================================
# CELERPLEX
# ============================================================

class CelerPlex():

    def __init__(
        self,
        df: pd.DataFrame,
        target: str,
        feature: list
    ):

        self.df = df
        self.target = target
        self.feature = feature

        self.best_cols = None
        self.model = None
        self.metrics = None

    def Classify(self, custom_house=None):

        model = DecisionTreeClassifier(
            random_state=42
        )

        encoded_df = pd.get_dummies(
            self.df[self.feature],
            drop_first=True,
            dtype=int
        )

        model, xtr, xtst, ytr, yts, best_cols = SmartData(
            self.df,
            model,
            encoded_df,
            self.df[self.target]
        )

        self.model = model
        self.best_cols = best_cols

        # ==========================================
        # PREDICTION DU DATASET DE TEST
        # ==========================================

        test_preds = model.predict(xtst)

        acc = accuracy_score(
            yts,
            test_preds
        )

        self.metrics = evaluate(
            yts,
            test_preds,
            "classification"
        )

        # ==========================================
        # PREDICTION PERSONNALISÉE
        # ==========================================

        if custom_house is not None:

            custom_df = pd.DataFrame(
                [custom_house],
                columns=self.feature
            )

            custom_df = pd.get_dummies(
                custom_df,
                drop_first=True,
                dtype=int
            )

            # Même structure que les données d'entraînement
            custom_df = custom_df.reindex(
                columns=encoded_df.columns,
                fill_value=0
            )

            # Garder uniquement les meilleures colonnes
            custom_df = custom_df[best_cols]

            pred = model.predict(
                custom_df
            )

            return pred, acc, model

        return test_preds, acc, model


    def Regress(self, custom_house=None):

        model = DecisionTreeRegressor(
            random_state=42
        )

        encoded_df = pd.get_dummies(
            self.df[self.feature],
            drop_first=True,
            dtype=int
        )

        model, xtr, xtst, ytr, yts, best_cols = SmartData(
            self.df,
            model,
            encoded_df,
            self.df[self.target]
        )

        self.model = model
        self.best_cols = best_cols

        print(
            "BEST COLUMNS :",
            best_cols
        )

        # ==========================================
        # PREDICTION DU DATASET DE TEST
        # ==========================================

        test_preds = model.predict(
            xtst
        )

        acc = r2_score(
            yts,
            test_preds
        )

        self.metrics = evaluate(
            yts,
            test_preds,
            "regression"
        )

        # ==========================================
        # PREDICTION PERSONNALISÉE
        # ==========================================

        if custom_house is not None:

            custom_df = pd.DataFrame(
                [custom_house],
                columns=self.feature
            )

            custom_df = pd.get_dummies(
                custom_df,
                drop_first=True,
                dtype=int
            )

            # Même structure que les données d'entraînement
            custom_df = custom_df.reindex(
                columns=encoded_df.columns,
                fill_value=0
            )

            # Garder uniquement les meilleures colonnes
            custom_df = custom_df[best_cols]

            pred = model.predict(
                custom_df
            )

            return pred, acc, model

        return test_preds, acc, model

# ============================================================
# ACCURATUS — XGBOOST
# ============================================================

class Accuratus():

    def __init__(self, df: pd.DataFrame, target: str, feature: list):

        self.df = df
        self.target = target
        self.feature = feature

    def Classify(self, custom_house=None):

        model = XGBClassifier(
            max_depth=10,
            learning_rate=0.4,
            n_estimators=100,
            random_state=42
        )

        encoded_df = pd.get_dummies(
            self.df[self.feature],
            drop_first=True,
            dtype=int
        )

        model, xtr, xtst, ytr, yts, best_cols = SmartData(
            self.df,
            model,
            encoded_df,
            self.df[self.target]
        )

        if custom_house is not None:

            test_preds = model.predict(xtst)
            acc = accuracy_score(yts, test_preds)
            model.metrics = evaluate(yts, test_preds, "classification")

            custom_df = pd.DataFrame(
                [custom_house],
                columns=self.feature
            )

            custom_df = pd.get_dummies(
                custom_df,
                drop_first=True,
                dtype=int
            )

            custom_df = custom_df.reindex(
                columns=encoded_df.columns,
                fill_value=0
            )

            pred = model.predict(custom_df[best_cols])

            return pred, acc, model

        pred = model.predict(xtst)
        acc = accuracy_score(yts, pred)
        model.metrics = evaluate(yts, pred, "classification")

        return pred, acc, model

    def Regress(self, custom_house=None):

        model = XGBRegressor(
            max_depth=10,
            learning_rate=0.4,
            n_estimators=100,
            random_state=42
        )

        encoded_df = pd.get_dummies(
            self.df[self.feature],
            drop_first=True,
            dtype=int
        )

        model, xtr, xtst, ytr, yts, best_cols = SmartData(
            self.df,
            model,
            encoded_df,
            self.df[self.target]
        )

        if custom_house is not None:

            test_preds = model.predict(xtst)
            acc = r2_score(yts, test_preds)
            model.metrics = evaluate(yts, test_preds, "regression")

            custom_df = pd.DataFrame(
                [custom_house],
                columns=self.feature
            )

            custom_df = pd.get_dummies(
                custom_df,
                drop_first=True,
                dtype=int
            )

            custom_df = custom_df.reindex(
                columns=encoded_df.columns,
                fill_value=0
            )

            pred = model.predict(custom_df[best_cols])

            return pred, acc, model

        pred = model.predict(xtst)
        acc = r2_score(yts, pred)
        model.metrics = evaluate(yts, pred, "regression")

        return pred, acc, model


# ============================================================
# VELOCITAS — LIGHTGBM
# ============================================================

class Velocitas():

    def __init__(self, df: pd.DataFrame, target: str, feature: list):

        self.df = df
        self.target = target
        self.feature = feature

    def Classify(self, custom_house=None):

        model = LGBMClassifier(
            max_depth=10,
            learning_rate=0.4,
            n_estimators=100,
            random_state=42,
            verbose=-1
        )

        encoded_df = pd.get_dummies(
            self.df[self.feature],
            drop_first=True
        )

        encoded_df.columns = [
            str(col)
            .replace(' ', '_')
            .replace('[', '')
            .replace(']', '')
            for col in encoded_df.columns
        ]

        model, xtr, xtst, ytr, yts, best_cols = SmartData(
            self.df,
            model,
            encoded_df,
            self.df[self.target]
        )

        if custom_house is not None:

            test_preds = model.predict(xtst)
            acc = accuracy_score(yts, test_preds)
            model.metrics = evaluate(yts, test_preds, "classification")

            custom_df = pd.DataFrame(
                [custom_house],
                columns=self.feature
            )

            custom_df = pd.get_dummies(
                custom_df,
                drop_first=True
            )

            custom_df.columns = [
                str(col)
                .replace(' ', '_')
                .replace('[', '')
                .replace(']', '')
                for col in custom_df.columns
            ]

            custom_df = custom_df.reindex(
                columns=encoded_df.columns,
                fill_value=0
            )

            pred = model.predict(custom_df[best_cols])

            return pred, acc, model

        pred = model.predict(xtst)
        acc = accuracy_score(yts, pred)
        model.metrics = evaluate(yts, pred, "classification")

        return pred, acc, model

    def Regress(self, custom_house=None):

        model = LGBMRegressor(
            max_depth=10,
            learning_rate=0.4,
            n_estimators=100,
            random_state=42,
            verbose=-1
        )

        encoded_df = pd.get_dummies(
            self.df[self.feature],
            drop_first=True
        )

        encoded_df.columns = [
            str(col)
            .replace(' ', '_')
            .replace('[', '')
            .replace(']', '')
            for col in encoded_df.columns
        ]

        model, xtr, xtst, ytr, yts, best_cols = SmartData(
            self.df,
            model,
            encoded_df,
            self.df[self.target]
        )

        if custom_house is not None:

            test_preds = model.predict(xtst)
            acc = r2_score(yts, test_preds)
            model.metrics = evaluate(yts, test_preds, "regression")

            custom_df = pd.DataFrame(
                [custom_house],
                columns=self.feature
            )

            custom_df = pd.get_dummies(
                custom_df,
                drop_first=True
            )

            custom_df.columns = [
                str(col)
                .replace(' ', '_')
                .replace('[', '')
                .replace(']', '')
                for col in custom_df.columns
            ]

            custom_df = custom_df.reindex(
                columns=encoded_df.columns,
                fill_value=0
            )

            pred = model.predict(custom_df[best_cols])

            return pred, acc, model

        pred = model.predict(xtst)
        acc = r2_score(yts, pred)
        model.metrics = evaluate(yts, pred, "regression")

        return pred, acc, model


# ============================================================
# EPTUS
# ============================================================

class Eptus():

    def __init__(
        self,
        df: pd.DataFrame,
        target: str,
        feature: list
    ):

        self.df = df.copy()
        self.target = target
        self.feature = feature

        self.df, _ = self.SmartCleaning()

        self.size = self.df.shape[0] + self.df.shape[1]
        self.content = self.df[self.target].nunique()

        self.model = None
        self.score = None
        self.name_model = None

    def SmartCleaning(self):

        self.df.dropna(inplace=True)

        if self.df.duplicated().any():

            self.df.drop_duplicates(inplace=True)

        detector = IsolationForest(
            contamination=0.05,
            random_state=42
        )

        print("========== DEBUG ==========")
        print("DF SHAPE :", self.df.shape)
        print("DF COLUMNS :", self.df.columns.tolist())
        print("FEATURES :", self.feature)
        print("DF FEATURES :")
        print(self.df[self.feature].head())
        print("DF FEATURES SHAPE :", self.df[self.feature].shape)
        print("===========================")

        encoded_df = pd.get_dummies(
            self.df[self.feature],
            drop_first=True,
            dtype=int
        )

        detector.fit(encoded_df)

        score_outlier = detector.score_samples(
            encoded_df
        )

        lignes_a_supprimer = []

        for index, score in enumerate(score_outlier):

            valeur_positive = abs(score)

            calc_probabilite = (
                (valeur_positive - 0.4)
                /
                (0.9 - 0.4)
            ) * 100

            calc_probabilite = max(
                0,
                min(100, calc_probabilite)
            )

            if calc_probabilite > 50:
                pass

        self.df.drop(
            self.df.index[lignes_a_supprimer],
            inplace=True
        )

        return self.df, self.target

    def CheckContent(self):

        if self.size <= 50000:

            return CelerPlex(
                self.df,
                self.target,
                self.feature
            )

        if self.size <= 100000:

            return Accuratus(
                self.df,
                self.target,
                self.feature
            )

        else:

            return Velocitas(
                self.df,
                self.target,
                self.feature
            )

    def Regress(self, custom_house=None):

        model = self.CheckContent()

        pred, accu, model = model.Regress(custom_house)

        return pred, accu, model

    def Classify(self, custom_house=None):

        model = self.CheckContent()

        pred, accu, model = model.Classify(custom_house)

        self.score = model.metrics

        return pred, accu, model
    
    def return_model_and_data(self):

        return self.model, 0, self.name_model

    def DictWeight(self, model, columns):
        weights = {}
    
        for column, importance in zip(columns, model.feature_importances_):
            weights[column] = float(importance)
    
        return weights