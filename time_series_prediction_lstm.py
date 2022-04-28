import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection
from keras import models, layers
from matplotlib import pyplot as plt
"""
        Input and output variables to choose from:
        Tboil: boiler temperature at each time, K
        Eturb: turbine energy at each time, kJ
        Pout: energy taken from turbine, kJ
        omega: turbine rotation speed, rad/s
        x1: water quality at boiler outlet
        h1: water enthalpy at boiler outlet, kJ/kg
        T1: water temperature at boiler outlet, K
        P1: water pressure at boiler outlet, MPa
        rho1: water density at boiler outlet, kg/m^3
        x2: water quality at turbine outlet
        h2: water enthalpy at turbine outlet, kJ/kg
        T2: water temperature at turbine outlet, K
        P2: water pressure at turbine outlet, MPa
        rho2: water density at turbine outlet, kg/m^3
"""

def load_data_npy(input_variables, output_variable):
    training_data = pd.DataFrame()
    for feature in [output_variable] + input_variables:
        training_data[feature] = pd.DataFrame(np.load('lstm_data/' + feature + '_log.csv.npy'))
    return training_data


def load_data(input_variables, output_variable):
    training_data = pd.DataFrame()
    for feature in [output_variable] + input_variables:
        training_data = pd.concat([training_data, pd.read_csv('transient_pred/' + feature + '_log.csv', nrows=1)])
        training_data.rename(index={0: feature}, inplace=True)
    training_data = training_data.T


    return training_data.drop(training_data.tail(1).index)


if __name__ == "__main__":
    # Configure inputs and outputs
    input_variables = ['Tboil','Pout', 'T1', 'T2']
    output_variable = 'Eturb'

    # Loads df with input columns followed by output column
    training_data = load_data_npy(input_variables, output_variable)
    Y_plot = training_data[output_variable]
    Y_plot = np.array(Y_plot)
    # Normalizes input data
    normalizer = preprocessing.StandardScaler()
    normalizer = normalizer.fit(training_data)
    training_data_norm = normalizer.transform(training_data)

    # Reshapes training data into LSTM friendly tensors based on look-back
    look_back = 7
    X, Y = [], []
    for i in range(look_back, len(training_data_norm)):
        X.append(training_data_norm[i - look_back:i, 0:training_data.shape[1]])
        Y.append(training_data_norm[i, 0])
    X, Y = np.array(X), np.array(Y)

    # Splits data into testing and training set
    test_size = 0.35
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, shuffle=False)

    # Now that data is prepared, we can make model
    '''
    Since relationship is observed to be simple, a simple architecture of LSTM - droput - dense is used
    dropout is included to prevent overfitting as is standard procedure with models (like LSTM) that are
    prone to it.
    '''
    model = models.Sequential()
    model.add(layers.LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
    #model.add(layers.LSTM(32, activation='relu', return_sequences=False))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mse')
    print(model.summary())
    hist = model.fit(X_train, Y_train, epochs=50, batch_size=25, validation_split=0.1, verbose=1)

    forecast_normalized = model.predict(X_test)
    forecast_normalized_repeat = np.repeat(forecast_normalized, training_data.shape[1], axis=-1)
    forecast = normalizer.inverse_transform(forecast_normalized_repeat)[:, 0]
    eval = model.evaluate(X_test, Y_test)
    print(eval)

    Y_test = Y_plot[-Y_test.shape[0]:]

    plot_num = 5000
    x_axis = [i for i in range(plot_num)]
    #plt.plot(x_axis, forecast_normalized[:plot_num])
    plt.plot(x_axis, Y_test[:plot_num], 'b-', label="Actual")
    plt.plot(x_axis, forecast[:plot_num], 'r-', label="Predicted")
    plt.xlabel("Time (s)")
    plt.ylabel("Energy (kJ)")
    plt.title("Turbine Energy vs Time")
    plt.legend()
    plt.show()
