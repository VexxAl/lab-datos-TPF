import pandas as pd
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler



class TimeSeriesFeatureEngineerIndex(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        target_col="Frio (Kw)",
        lags=(1, 2, 3),
        ma_windows=(3, 7)
    ):
        self.target_col = target_col
        self.lags = lags
        self.ma_windows = ma_windows

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Aseguramos que el índice sea datetime
        X.index = pd.to_datetime(X.index)

        # ------- LAGS de Frio (Kw) -------
        for lag in self.lags:
            col_name = f"{self.target_col}_lag_{lag}"
            X[col_name] = X[self.target_col].shift(lag)

        # ------- Promedios móviles de Frio (Kw) -------
        for window in self.ma_windows:
            col_name = f"{self.target_col}_ma_{window}"
            X[col_name] = (
                X[self.target_col]
                .rolling(window=window, min_periods=1)
                .mean()
            )

        # ------- Features de calendario desde el índice -------
        # 0 = lunes, 6 = domingo
        X["day_of_week"] = X.index.dayofweek
        X["month"] = X.index.month
        X["is_weekend"] = X["day_of_week"].isin([5, 6]).astype(int)

        return X

class ImputationSelector(BaseEstimator, TransformerMixin):
    def __init__(self, method="knn"):
        """
        method: 'knn', 'linear', 'mice'
        """
        self.method = method

    def fit(self, X, y=None):
        # Guardar columnas e índice
        self.columns_ = X.columns
        self.index_ = X.index
        
        # Elegir imputador
        if self.method == "knn":
            self.imputer_ = KNNImputer(n_neighbors=5)

        elif self.method == "linear":
            self.imputer_ = IterativeImputer(
                estimator=LinearRegression(),
                max_iter=20,
                random_state=42
            )

        elif self.method == "mice":
            self.imputer_ = IterativeImputer(
                estimator=BayesianRidge(),
                max_iter=20,
                random_state=42
            )

        # Ajustar imputador (recibe numpy, pero X es DataFrame y funciona igual)
        self.imputer_.fit(X)

        return self

    def transform(self, X):
        # Asegurar que X sea DataFrame con columnas correctas (evita problemas dentro del Pipeline)
        X = pd.DataFrame(X, columns=self.columns_)

        # Imputación
        X_imp = self.imputer_.transform(X)

        # Devolver DataFrame con columnas e índice originales
        return pd.DataFrame(X_imp, columns=self.columns_, index=X.index)

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        # opcional: podrías validar que todas existan
        self.feature_names_ = self.feature_names
        return self

    def transform(self, X):
        # aseguramos DataFrame
        X = pd.DataFrame(X)
        return X[self.feature_names_]
    
class ImputationSelector(BaseEstimator, TransformerMixin):
    def __init__(self, method="knn"):
        """
        method: 'knn', 'linear', 'mice'
        """
        self.method = method

    def fit(self, X, y=None):
        # Guardar columnas e índice
        self.columns_ = X.columns
        self.index_ = X.index
        
        # Elegir imputador
        if self.method == "knn":
            self.imputer_ = KNNImputer(n_neighbors=5)

        elif self.method == "linear":
            self.imputer_ = IterativeImputer(
                estimator=LinearRegression(),
                max_iter=20,
                random_state=42
            )

        elif self.method == "mice":
            self.imputer_ = IterativeImputer(
                estimator=BayesianRidge(),
                max_iter=20,
                random_state=42
            )

        # Ajustar imputador (recibe numpy, pero X es DataFrame y funciona igual)
        self.imputer_.fit(X)

        return self

    def transform(self, X):
        # Asegurar que X sea DataFrame con columnas correctas (evita problemas dentro del Pipeline)
        X = pd.DataFrame(X, columns=self.columns_)

        # Imputación
        X_imp = self.imputer_.transform(X)

        # Devolver DataFrame con columnas e índice originales
        return pd.DataFrame(X_imp, columns=self.columns_, index=X.index)

class LOFOutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, method="winsorize", std_factor=3):
        """
        method: 'winsorize' o 'clip'
        std_factor: número de desvíos estándar para clipping
        """
        self.method = method
        self.std_factor = std_factor
        self.columns_ = None

    def fit(self, X, y=None):
        # Aseguramos DataFrame
        X = pd.DataFrame(X)
        self.columns_ = X.columns

        # Guardamos también los nombres como hace sklearn
        self.feature_names_in_ = self.columns_

        # Entrenamos LOF solo para detectar outliers (por ahora solo guardamos la máscara)
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
        self.outlier_mask_ = lof.fit_predict(X) == -1

        # Estadísticos por columna
        self.means_ = X.mean()
        self.stds_ = X.std()
        self.p1_ = X.quantile(0.01)
        self.p99_ = X.quantile(0.99)

        return self

    def transform(self, X):
        # Volvemos a DataFrame con mismas columnas
        X = pd.DataFrame(X, columns=self.columns_)

        if self.method == "winsorize":
            # Winsorización p1–p99 por columna
            X = X.clip(lower=self.p1_, upper=self.p99_, axis=1)

        elif self.method == "clip":
            # Clipping por medias ± k*std por columna
            lower = self.means_ - self.std_factor * self.stds_
            upper = self.means_ + self.std_factor * self.stds_
            X = X.clip(lower=lower, upper=upper, axis=1)

        # Devolvemos DataFrame con mismo índice y columnas
        return X

class SkewTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, method="none"):
        self.method = method

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.feature_names_in_ = X.columns
        self.num_cols_ = X.select_dtypes(include=[np.number]).columns

        if self.method == "yeojohnson":
            self.pt_ = PowerTransformer(method="yeo-johnson", standardize=False)
            self.pt_.fit(X[self.num_cols_])

        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.feature_names_in_)

        if self.method == "none":
            return X

        if self.method == "yeojohnson":
            X_num = X[self.num_cols_]
            X[self.num_cols_] = self.pt_.transform(X_num)

        return X

best_params_preproc = {'imputer': 'linear', 'outliers': 'clip', 'std_factor': 4.378253933658169, 'skew_method': 'yeojohnson'}
best_features = ['Frio (Kw)', 'Frio (Kw)_ma_3', 'Frio (Kw)_ma_14', 'Sala Maq (Kw)', 'Frio (Kw)_ma_7', 'Servicios (Kw)', 'KW Obrador Contratistas', 'Frio (Kw)_lag_2', 'Tot  A130/330/430', 'KW Servicio L2']

best_imputer = best_params_preproc["imputer"]       
best_outliers = best_params_preproc["outliers"]     
best_std_factor = best_params_preproc["std_factor"]
best_skew_method = best_params_preproc["skew_method"]   


full_preprocess_pipe = Pipeline([
    ("imputer", ImputationSelector(method=best_imputer)),
    ("outliers", LOFOutlierHandler(method=best_outliers, std_factor=best_std_factor)),
    ("time_features", TimeSeriesFeatureEngineerIndex(
        target_col="Frio (Kw)",
        lags=(1, 2, 3, 24),
        ma_windows=(3, 7, 14),
    )),
    ("feature_selector", FeatureSelector(best_features)),
    ("skew", SkewTransformer(method=best_skew_method)),
    ("scaler", StandardScaler()),    
])