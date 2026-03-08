import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Input
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import kline_fetccher
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


class ComprehensiveFundPredictor:

    def __init__(self, fund_code):
        self.fetcher = kline_fetccher.KlineFetcher()
        self.fund_code = fund_code

        self.data = None
        self.enhanced_data = None

        self.model = None
        self.scaler = None

        self.topic = "close"

    def load_and_prepare_data(self, time_step=10):
        self.data = self.fetcher.fetch_data(self.fund_code)
        if len(self.data) < time_step:
            print(
                f"error: data length {len(self.data)} is less than time_step {time_step}"
            )
            return False

        self.enhanced_data = self.data.copy()

        feature_columns = [col for col in self.enhanced_data.columns if col != "date"]
        self.enhanced_data = self.enhanced_data[feature_columns]

        print(f"already have {self.enhanced_data.columns} features")
        self.print_feature_explanations()
        return True

    def build_advanced_model(self, model_type="lstm", time_step=10, **kwargs):
        X, y = self._prepare_sequences(time_step=time_step)
        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        self.model = self._build_lstm_model((X_train.shape[1], X_train.shape[2]))
        return X_train, y_train, X_test, y_test

    def _build_lstm_model(self, input_shape):
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(LSTM(units=32, return_sequences=False, activation="tanh"))
        model.add(Dense(units=16, activation="relu"))
        model.add(Dense(units=1, activation="linear"))
        model.summary()
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")
        return model

    def _prepare_sequences(self, time_step=10):
        # create scaler for all features
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = self.scaler.fit_transform(self.enhanced_data)

        # create separate scaler for close column only
        self.close_scaler = MinMaxScaler(feature_range=(0, 1))
        close_data = self.enhanced_data[[self.topic]].values
        self.close_scaler.fit(close_data)

        X = []
        y = []
        tip = self.enhanced_data.columns.get_loc(self.topic)
        for i in range(time_step, len(scaled_data)):
            X.append(scaled_data[i - time_step : i])
            y.append(scaled_data[i, tip])
        return np.array(X), np.array(y)

    def train_model(self, X_train, y_train, epochs=300, batch_size=16):
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6),
        ]
        return self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=callbacks,
        )

    def calculate_accuracy(self, y_true, y_pred, threshold_factor=3):
        nav_std = self.data[self.topic].std()
        threshold = nav_std * threshold_factor
        diff = np.abs(y_true - y_pred)
        correct = np.sum(diff < threshold)
        accuracy = correct / len(y_true)
        return accuracy, threshold

    def evaluate_model(self, X, y):
        y_pred = self.model.predict(X)
        y_true = self._inverse_transform(y)
        y_pred = self._inverse_transform(y_pred)
        accuracy, threshold = self.calculate_accuracy(y_true, y_pred)
        print(f"accuracy: {accuracy:.4f}, threshold: {threshold:.6f}")
        return {
            "y_true": y_true,
            "y_pred": y_pred,
        }

    def plot_predictions(self, evaluation_results, future_predictions, time_step=10):
        plt.rcParams["axes.unicode_minus"] = False

        train_true = evaluation_results["train_true"]
        train_pred = evaluation_results["train_pred"]
        test_true = evaluation_results["test_true"]
        test_pred = evaluation_results["test_pred"]

        plt.figure(figsize=(12, 8))

        # combine train and test actual values for continuous line
        all_actual = np.concatenate([train_true, test_true])
        all_actual_x = range(time_step, time_step + len(all_actual))

        # plot continuous actual line (blue solid)
        plt.plot(all_actual_x, all_actual, label="actual", color="blue", linewidth=1.5)

        # train predict (green dashed)
        train_x = range(time_step, time_step + len(train_true))
        plt.plot(
            train_x,
            train_pred,
            label="train predict",
            color="green",
            linestyle="--",
            linewidth=1.5,
        )

        # test predict (orange dashed)
        test_start = time_step + len(train_true)
        test_x = range(test_start, test_start + len(test_true))
        plt.plot(
            test_x,
            test_pred,
            label="test predict",
            color="orange",
            linestyle="--",
            linewidth=1.5,
        )

        # future predictions (red solid)
        future_start = test_start + len(test_true)
        future_x = range(future_start, future_start + len(future_predictions))
        plt.plot(
            future_x,
            future_predictions,
            label="futurex predict",
            color="red",
            linewidth=1.5,
        )

        plt.title(
            f"{self.fund_code} Close Validation and Future Predictions", fontsize=16
        )
        plt.xlabel("Date Index", fontsize=12)
        plt.ylabel("Close Price", fontsize=12)
        plt.legend(fontsize=12, loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def _inverse_transform(self, data):
        # use close_scaler for inverse transform of close column
        return self.close_scaler.inverse_transform(data.reshape(-1, 1)).flatten()

    def predict_future(self, days=5, time_step=10):
        scaled_data = self.scaler.transform(self.enhanced_data)
        if len(scaled_data) < time_step:
            print(
                f"predict future failed, data length {len(scaled_data)} is less than time_step {time_step}"
            )
            return None
        last_sequence = scaled_data[-time_step:]

        predictions = []
        current_sequence = last_sequence.copy()

        for _ in range(days):
            pred = self.model.predict(current_sequence.reshape(1, time_step, -1))[0][0]
            predictions.append(pred)
            new_row = current_sequence[-1].copy()
            new_row[0] = pred
            current_sequence = np.vstack([current_sequence[1:], new_row])

        predictions = np.array(predictions)
        predictions_original = self._inverse_transform(predictions)

        print("future predictions:")
        for i, pred in enumerate(predictions_original, 1):
            print(f"day {i}: {pred:.4f}")

        return predictions_original

    def print_feature_explanations(self):
        # TODO 打印特征名称和详细解释
        pass

    def analyze_holdings(self):
        """分析基金持仓"""
        print("\n分析基金持仓...")

        try:
            if self.holdings_analyzer:
                # 行业配置分析
                sector_allocation = self.holdings_analyzer.analyze_sector_allocation()
                # 投资风格分析
                style_analysis = self.holdings_analyzer.analyze_style()
                # 市值分布分析
                cap_analysis = self.holdings_analyzer.analyze_market_cap()
                # 前10大重仓股
                top_holdings = self.holdings_analyzer.get_top_holdings(10)
                # 持仓集中度
                concentration = self.holdings_analyzer.calculate_concentration()
                return {
                    "sector_allocation": sector_allocation,
                    "style_analysis": style_analysis,
                    "cap_analysis": cap_analysis,
                    "top_holdings": top_holdings,
                    "concentration": concentration,
                }
            else:
                print("基金持仓分析模块不可用")
                return {}
        except Exception as e:
            print(f"分析基金持仓时出错: {e}")
            return {}

    def generate_comprehensive_report(self):
        print("\n" + "=" * 60)

        print(f"Code:\t\t{self.fund_code}")
        print(f"date range:\t{self.data['date'].min()} ~ {self.data['date'].max()}")
        print(f"records:\t{len(self.data)}")
        print(f"features:\t{len(self.enhanced_data.columns)}")

        print(f"Close:\t{self.data['close'].iloc[-1]:.4f}")
        print(f"high:\t{self.data['close'].max():.4f}")
        print(f"low:\t{self.data['close'].min():.4f}")
        print(f"avg:\t{self.data['close'].mean():.4f}")

        print("\n" + "=" * 60)


def main(
    fund_code="017057",
    model_type="lstm",
    predict_days=5,
    time_step=10,
    epochs=300,
    batch_size=16,
):
    predictor = ComprehensiveFundPredictor(fund_code)
    if not predictor.load_and_prepare_data(time_step=time_step):
        print("load data failed")
        return None

    X_train, y_train, X_test, y_test = predictor.build_advanced_model(
        model_type=model_type, time_step=time_step
    )

    predictor.train_model(X_train, y_train, epochs=epochs, batch_size=batch_size)

    print("predict train set:")
    tran_results = predictor.evaluate_model(X_train, y_train)
    print("predict test set:")
    test_results = predictor.evaluate_model(X_test, y_test)

    future_predictions = predictor.predict_future(
        days=predict_days, time_step=time_step
    )

    predictor.plot_predictions(
        {
            "train_true": tran_results["y_true"],
            "train_pred": tran_results["y_pred"],
            "test_true": test_results["y_true"],
            "test_pred": test_results["y_pred"],
        },
        future_predictions,
        time_step=time_step,
    )

    predictor.generate_comprehensive_report()

    return predictor


if __name__ == "__main__":
    main("159625")
