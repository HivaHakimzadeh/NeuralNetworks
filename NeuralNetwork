import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split



class NeuralNet:
    def __init__(self, dataFile, header=True):
        self.raw_input = pd.read_csv(dataFile)

    def preprocess(self):
      # Pre-processing data
      # There are no missing values also all the feature variables are numerical so there is no need
      # to convert
        self.processed_data = self.raw_input
        return 0

    def train_evaluate(self):
        # Define hyperparameter combinations to try
        ncols = len(self.processed_data.columns)
        nrows = len(self.processed_data.index)
        X = self.processed_data.iloc[:, 0:(ncols - 1)]
        Y = self.processed_data.iloc[:, (ncols-1)]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)
        dataset = {
            'X_train': X_train,
            'Y_train': Y_train,
            'X_test': X_test,
            'Y_test': Y_test
        }
        hyperparameter = [
            {'hidden_layers_count': 3, 'nodes_count': 64},
            {'hidden_layers_count': 3, 'nodes_count': 32},
        ]

        # Create empty lists to store results

        learning_rate = [0.01, 0.1]
        max_iterations = [100, 200] # also known as epochs
        activations = ['sigmoid', 'tanh', 'relu']

        # Iterate over hyperparameter combinations
        models = []
        histories = []
        results = []
        for i in activations:
            for params in hyperparameter:
                for iters in max_iterations:
                    for ir in learning_rate:
                        # Build neural network model
                        model = Sequential()
                        for _ in range(params['hidden_layers_count']):
                            model.add(Dense(params['nodes_count'], activation= i ))
                        model.add(Dense(1))
                        model.compile(optimizer=Adam(learning_rate = ir), loss='mse', metrics=['accuracy'])

                        # Train model
                        history = model.fit(dataset['X_train'], dataset['Y_train'], epochs= iters, validation_data=(dataset['X_test'], dataset['Y_test']))

                        # Evaluate model
                        train_acc = model.evaluate(dataset['X_train'], dataset['Y_train'])[1]
                        test_acc = model.evaluate(dataset['X_test'], dataset['Y_test'])[1]

                        # Record results
                        models.append(model)
                        histories.append(history)
                        results.append({
                            'activation': i,
                            'hidden_layers_count': params['hidden_layers_count'],
                            'nodes_count': params['nodes_count'],
                            'Training Accuracy': train_acc,
                            'Test Accuracy': test_acc,
                            'Training MSE': mean_squared_error(dataset['Y_train'], model.predict(dataset['X_train'])),
                            'Test MSE': mean_squared_error(dataset['Y_test'], model.predict(dataset['X_test'])),
                            'Epochs': iters,
                            'Learning Rate': ir
                        })

            # Plot model
            figsize = (20, 20)
            plt.figure(figsize=figsize)
            for history in histories:
                plt.plot(history.history['accuracy'], label='Training Accuracy')
                plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.title('Performance: Accuracy X Epochs '+ i)
            plt.legend()
            plt.show()

            # Print results table
            print(tabulate(results, headers='keys', tablefmt='psql'))

            models.clear()
            histories.clear()
            results.clear()


if __name__ == '__main__':
    neural_network = NeuralNet('https://raw.githubusercontent.com/77EminSarac77/Dataset-for-Linear-Regression/main/Diabetespred.csv')
    neural_network.preprocess()
    neural_network.train_evaluate()
