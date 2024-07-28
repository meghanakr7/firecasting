import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor
from pytorch_tabnet.tab_model import TabNetRegressor
import pickle
import warnings

warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")


class ModelHandler:
    def fit(self, X_train, y_train):
        raise NotImplementedError

    def predict(self, X_test):
        raise NotImplementedError

    def save_model(self, model, model_path):
        raise NotImplementedError

    def load_model(self, model_path):
        raise NotImplementedError


class LightGBMHandler(ModelHandler):
    def __init__(self):
        self.model = LGBMRegressor(n_jobs=-1, random_state=42)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def save_model(self, model_path):
        with open(model_path, 'wb') as model_file:
            pickle.dump(self.model, model_file)

    def load_model(self, model_path):
        with open(model_path, 'rb') as model_file:
            self.model = pickle.load(model_file)


class TabNetHandler(ModelHandler):
    def __init__(self):
        self.model = TabNetRegressor(
            n_d=17,
            n_a=41,
            n_steps=4,
            gamma=1.1546672563068268,
            lambda_sparse=0.00042602006758391
        )

    def fit(self, X_train, y_train):
        y_train = y_train.to_numpy().reshape(-1, 1)
        self.model.fit(
            X_train.values, y_train,
            eval_set=[(X_train.values, y_train)],
            eval_metric=['mae'],
            max_epochs=100,
            patience=10,
            batch_size=256,
            virtual_batch_size=128,
        )

    def predict(self, X_test):
        return self.model.predict(X_test.values)

    def save_model(self, model_path):
        self.model.save_model(model_path)

    def load_model(self, model_path):
        self.model.load_model(model_path)
    


class WildfireModelTrainer:
    def __init__(self, model_type="lightgbm", chosen_input_columns=[], training_data_folder="/path/to/training/data"):
        self.training_data_folder = training_data_folder
        self.target_col = 'FRP'
        self.chosen_input_columns = chosen_input_columns
        self.model_handlers = {
            'large_west': self.init_model_handler(model_type),
            'small_west': self.init_model_handler(model_type),
            'large_east': self.init_model_handler(model_type),
            'small_east': self.init_model_handler(model_type),
        }

    def init_model_handler(self, model_type):
        if model_type == "tabnet":
            return TabNetHandler()
        else:
            return LightGBMHandler()

    def read_original_txt_files(self, folder_path, datestr):
        file_path = os.path.join(folder_path, f"firedata_{datestr}.txt")
        print(f"Reading original file: {file_path}")
        return pd.read_csv(file_path)

    def get_one_day_time_series_training_data(self, folder_path, target_day):
        df = self.read_original_txt_files(folder_path, target_day)
        target_dt = datetime.strptime(target_day, '%Y%m%d')
        for i in range(7):
            past_dt = target_dt - timedelta(days=i + 1)
            past_df = self.read_original_txt_files(folder_path, past_dt.strftime('%Y%m%d'))
            for c in ['FWI', 'VPD', 'P', 'FRP']:
                df[f'{c}_{i + 1}_days_ago'] = past_df[c]
        return df

    def prepare_training_data(self, folder_path, target_date):
        if not os.path.exists(self.training_data_folder):
            os.makedirs(self.training_data_folder)
            print(f"Folder created: {self.training_data_folder}")
        else:
            print(f"Folder already exists: {self.training_data_folder}")

        train_file_path = os.path.join(self.training_data_folder, f"{target_date}_time_series_with_new_window.csv")

        if os.path.exists(train_file_path):
            print(f"File {train_file_path} exists")
            df = pd.read_csv(train_file_path)
        else:
            df = self.get_one_day_time_series_training_data(folder_path, target_date)
            df.fillna(-999, inplace=True)
            df = df[(df[['FRP_1_days_ago', 'Nearest_1', 'Nearest_2', 'Nearest_3', 'Nearest_4', 'Nearest_5', 'Nearest_6', 'Nearest_7', 'Nearest_8']] > 0).any(axis=1)]
            df.to_csv(train_file_path, index=False)

        X = df[self.chosen_input_columns]
        y = df[self.target_col]
        return X, y

    def train_model(self, start_date_str, end_date_str, folder_path, model_paths, fire_size_threshold=300, region_dividing_longitude=-100):
        start_date = datetime.strptime(start_date_str, "%Y%m%d")
        end_date = datetime.strptime(end_date_str, "%Y%m%d")
        current_date = start_date

        # Initialize data containers
        all_data = {
            'large_west': [],
            'small_west': [],
            'large_east': [],
            'small_east': []
        }
        
        while current_date <= end_date:
            date_str = current_date.strftime("%Y%m%d")
            print(f"Processing data for {date_str}")

            # Prepare training data
            X, y = self.prepare_training_data(folder_path, date_str)

            if X.empty or y.empty:
                print(f"No data available for {date_str}. Skipping...")
                current_date += timedelta(days=1)
                continue

            X[self.target_col] = np.log10(y + 1e-2)

            # Determine if the fire is large or small
            is_large_fire = y > fire_size_threshold

            # Determine if the location is west or east
            is_west = X['LON'] < region_dividing_longitude

            # Append data to the appropriate category
            for i in range(len(y)):
                if is_west[i]:
                    if is_large_fire[i]:
                        all_data['large_west'].append(X.iloc[i])
                    else:
                        all_data['small_west'].append(X.iloc[i])
                else:
                    if is_large_fire[i]:
                        all_data['large_east'].append(X.iloc[i])
                    else:
                        all_data['small_east'].append(X.iloc[i])

            current_date += timedelta(days=1)

        # Debugging output
        for key in all_data:
            if not all_data[key]:
                print(f"No data for category: {key}")

        # Train models for each category
        for key in all_data:
            if all_data[key]:
                all_data_combined = pd.DataFrame(all_data[key])
                all_data_combined = all_data_combined.dropna(subset=[self.target_col])
                X = all_data_combined[self.chosen_input_columns]
                y = all_data_combined[self.target_col]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

                self.model_handlers[key].fit(X_train, y_train)
                y_pred_test = self.model_handlers[key].predict(X_test)

                mse = mean_squared_error(y_test, y_pred_test)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred_test)
                r2 = r2_score(y_test, y_pred_test)

                print(f"Category: {key}")
                print(f"Mean Squared Error (MSE): {mse}")
                print(f"Root Mean Squared Error (RMSE): {rmse}")
                print(f"Mean Absolute Error (MAE): {mae}")
                print(f"R-squared (R2): {r2}")

                # Save model for each category
                model_path = model_paths[key]
                self.model_handlers[key].save_model(model_path)
                print(f"Save to {model_path}")

                now = datetime.now()
                date_time = now.strftime("%Y%d%m%H%M%S")
                random_model_path = f"{model_path}_{start_date_str}_{end_date_str}_{date_time}.pkl"
                self.model_handlers[key].save_model(random_model_path)
                print(f"A copy of the model is saved to {random_model_path}")
            else:
                print(f"No data to train model for category: {key}")

        print("Training completed for all categories.")


# Define global variables that can be imported by others
model_type = "lightgbm"  # Can be 'lightgbm' or 'tabnet'
model_paths = {
    'large_west': f"/groups/ESS3/zsun/firecasting/model/fc_{model_type}_large_west.pkl",
    'small_west': f"/groups/ESS3/zsun/firecasting/model/fc_{model_type}_small_west.pkl",
    'large_east': f"/groups/ESS3/zsun/firecasting/model/fc_{model_type}_large_east.pkl",
    'small_east': f"/groups/ESS3/zsun/firecasting/model/fc_{model_type}_small_east.pkl"
}
folder_path = '/groups/ESS3/yli74/data/AI_Emis/firedata_VHI'
training_data_folder = "/groups/ESS3/zsun/firecasting/data/train/"
chosen_input_columns = [
    'FRP_1_days_ago', 'Nearest_1', 'Nearest_5', 'Nearest_7', 'Nearest_3', 'FRP_2_days_ago', 'VPD', 'V', 'LAT', 'LON', 'FWI', 'Nearest_17', 'Land_Use', 'RH'
]

if __name__ == "__main__":
    trainer = WildfireModelTrainer(
        model_type=model_type,
        training_data_folder=training_data_folder, 
        chosen_input_columns=chosen_input_columns
    )
    start_date_str = "20200109"
    end_date_str = "20200130"
    
    trainer.train_model(start_date_str, end_date_str, folder_path, model_paths, fire_size_threshold=1, region_dividing_longitude=-100)
    print(f"Training completed and models saved to {model_paths}")

