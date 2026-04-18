import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDRegressor
import joblib

class LinearRegressionModel:
    def __init__(self):
        pass

    def load_data(self, file_path):
        data = pd.read_csv(file_path)
        return data
    
    def drop_feature(self, data, feature_name):
        data = data.drop(column=[feature_name])
    
    def create_featur_label(self, data, feature_name):
        X = data.drop(columns=feature_name)
        y = data[feature_name]
        return X, y
    
    def train_test_split(self, X, y, test_size=0.2, random_state=42):
        return train_test_split(X,y, test_size=0.2, random_state=42)
    
    def standarized_data(self, X_train, X_test):
        numeric_features = ["SquareFeet","BedRooms", "Bathrooms"]
        categorical_features = ["Neighborhood"]

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(), categorical_features)
            ]
        )

        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)
        return X_test,X_train

    def train_model(self, X_train, y_train):
        sgd_model = SGDRegressor(loss="squared_error", max_iter=1000, learning_rate="invscalling", eta=0.01)
        sgd_model.fit(X_train, y_train)
        return sgd_model
    
if __name__ == "__main__":
    model = LinearRegressionModel()
    data = model.load_data("C:\Users\92324\Desktop\ML_from_Sratch\ML_from_Scratch\Mlops\data\housing_price_dataset.csv")
    data = model.drop_feature(data, "ID")
    X, y = model.create_featur_label(data, "Price")
    X_train, X_test, y_train, y_test =model.train_test_split(X,y)
    X_train = model.standarized_data(X_train, X_test)
    sgd_model = model.train_model(X_train, y_train)

    joblib.dump(sgd_model, "linear_regression_model.pkl")