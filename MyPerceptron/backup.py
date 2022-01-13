import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import random


# Parameters...
# Για να χρησιμοποιήσεται το dataset1.csv αλλάξτε τα παρακάτω...
# urlDataset = 'dataset1.csv'
# n_rows = 100
# n_inputs = 3
# act_outputs = [1, -1]
# Για 3D act_outputs = ['C1', 'C2']

urlDataset = 'dataset1.csv'  # Dataset URL
learning_rate = 1.5
n_rows = 100
n_inputs = 2    # Changes with the dataset
weights = random.sample(range(-2, 2), n_inputs + 1)  # +1 is for the included weight for the bias
bias = 1
seasons = 25

SelectedPredictions = np.zeros(n_rows)
Show_Seasons = [1, 3, 10, 18]
plotWeights = np.zeros(shape=(len(Show_Seasons), n_inputs +1))


# Activation Function Outputs [First , Second]. Changes according to the dataset that is being used
act_outputs = [1, -1]

# Set Target Label Name...
t_label = 'Y'
# Set Label Starting Character
input_label = 'X'


def initDataset(UrlDataset, n):
    data = pd.read_csv(UrlDataset, nrows=n)
    return data


# Returns 0 if the function of the linear output is greater or equal than 0
def activationFunc(AF_output):
    if isinstance(act_outputs[0], int):
        return np.where(AF_output >= 0, act_outputs[0], act_outputs[1])
    else:
        return np.where(AF_output >= 0, 1, 0)


# Load Data
dataset = initDataset(urlDataset, n_rows)
df_target = pd.DataFrame(dataset, columns=[t_label])
target = df_target.to_numpy()

if isinstance(act_outputs[0], str):
    target = np.zeros(n_rows)
    for col, row in df_target.iterrows():
        if row[t_label] == act_outputs[0]:
            target[col] = 1
        elif row[t_label] == act_outputs[1]:
            target[col] = 0

Iter = 0
dfX_train = pd.DataFrame()
for col in dataset:
    if col != t_label:
        Iter += 1
        label = input_label + str(Iter)
        dfX_train[Iter] = pd.DataFrame(dataset, columns=[label])
X_train = dfX_train.to_numpy()

predicted = np.zeros(n_rows)
current_season = 0
ShowIter = 0
while current_season < seasons:
    current_iter = 0
    for xi in X_train:
        linear_output = 0
        for i in range(0, n_inputs):
            linear_output += xi[i] * weights[i]

        linear_output += weights[n_inputs] * bias
        predicted_y = activationFunc(linear_output)

        if predicted_y != target[current_iter]:
            for i in range(0, n_inputs):
                weights[i] = weights[i] + learning_rate * (target[current_iter] - predicted_y) * xi[i]

            weights[n_inputs] = weights[n_inputs] + learning_rate * (target[current_iter] - predicted_y)
        else:
            predicted[current_iter] = predicted_y
        current_iter += 1
    if current_season in Show_Seasons:
        for i in range(0, n_inputs + 1):
            plotWeights[ShowIter][i] = weights[i]
        ShowIter += 1
    current_season += 1

if n_inputs == 2:
    print('W2(bias) =\t', weights[n_inputs], "\nW1 =\t\t", weights[0], "\nW2 =\t\t", weights[1])
    print("Linear Function Formula -> Linear_Output = W0*bias + W1*x1 + W2*x2\n", )
if n_inputs == 3:
    print('W3(bias) =\t', weights[n_inputs], "\nW0 = \t\t", weights[0], "\nW1 = \t\t", weights[1], "\nW2 = \t\t", weights[2])
    print("Linear Function Formula -> Linear_Output = W0*bias + W1*x1 + W2*x2 + W3*x3\n")

if n_inputs == 3:
    x = np.arange(0, np.amax(X_train[:, [0]]), 0.1)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(X_train[:, [0]], X_train[:, [1]], X_train[:, [2]], marker='o', c=predicted)

    y = np.arange(0, np.amax(X_train[:, [0]]), 0.1)

    x, y = np.meshgrid(x, y)
    z = (-weights[0] / weights[2]) * x + (-weights[1] / weights[2]) * y + (-weights[n_inputs] / weights[2])
    # Παρωμοίος και στο δυσδιάστατο επίπεδο αλλά η συνάρτηση εξαρτάτε από τους πίνακες x και y + το intercept του z
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm)
    ax.set(xlabel='X1', ylabel='X2', zlabel='X3')

final_target = ["" for x in range(n_rows)]
final_prediction = ["" for x in range(n_rows)]

# Conversion...
for i in range(0, n_rows):
    if target[i] == 1:
        final_target[i] = act_outputs[0]
    else:
        final_target[i] = act_outputs[1]

    if SelectedPredictions[i] == 1:
        final_prediction[i] = act_outputs[0]
    else:
        final_prediction[i] = act_outputs[1]

AllPredicted = True
countSuccess = 0
for i in range(0, n_rows):
    if target[i] != predicted[i]:
        AllPredicted = False
    else:
        countSuccess += 1
    print("Last Predicted Values: ", final_prediction[i], "\tExpected Value: ", final_target[i])
    # print("Last Predicted Values: ", predicted[i], "\tExpected Value: ", target[i])

if AllPredicted:
    print("All Training Data Predicted at ", countSuccess/n_rows*100, '% Accuracy')
if not AllPredicted:
    print("Not all training Data was Predicted!", countSuccess/n_rows*100, '% Accurate')


if n_inputs == 2:
    x = np.arange(0, np.amax(X_train[:, [0]]), 0.1)
    fig = plt.figure()
    # x = np.meshgrid(x)
    if len(Show_Seasons) >= 1:
        y = (-plotWeights[1][0] / plotWeights[1][1]) * x + (-plotWeights[1][n_inputs] / plotWeights[1][1])
        ax = fig.add_subplot()
        ax.scatter(X_train[:, [0]], X_train[:, [1]], marker='o', c=predicted)
        ax.set_xlim([min(X_train[:, [0]]) - 2, max(X_train[:, [0]]) + 2])
        ax.set_ylim([min(X_train[:, [1]]) - 2, max(X_train[:, [1]] + 2)])
        ax.plot(x, y, 'r')
        plt.title("Season " + str(Show_Seasons[1]))
        plt.draw()
        plt.pause(3)
        plt.clf()

    if len(Show_Seasons) >= 2:
        y = (-plotWeights[2][0] / plotWeights[2][1]) * x + (-plotWeights[2][n_inputs] / plotWeights[2][1])

        ax = fig.add_subplot()
        ax.scatter(X_train[:, [0]], X_train[:, [1]], marker='o', c=predicted)
        ax.set_xlim([min(X_train[:, [0]]) - 2, max(X_train[:, [0]]) + 2])
        ax.set_ylim([min(X_train[:, [1]]) - 2, max(X_train[:, [1]] + 2)])
        ax.plot(x, y, 'r')
        plt.title("Season " + str(Show_Seasons[2]))
        plt.draw()
        plt.pause(3)
        plt.clf()

    if len(Show_Seasons) >= 3:
        y = (-plotWeights[3][0] / plotWeights[3][1]) * x + (-plotWeights[3][n_inputs] / plotWeights[3][1])

        ax = fig.add_subplot()
        ax.scatter(X_train[:, [0]], X_train[:, [1]], marker='o', c=predicted)
        ax.set_xlim([min(X_train[:, [0]]) - 2, max(X_train[:, [0]]) + 2])
        ax.set_ylim([min(X_train[:, [1]]) - 2, max(X_train[:, [1]] + 2)])
        ax.plot(x, y, 'r')
        plt.title("Season " + str(Show_Seasons[3]))
        plt.draw()
        plt.pause(3)
        plt.clf()

    y = (-weights[0] / weights[1]) * x + (-weights[n_inputs] / weights[1])
    # Η πάνω μορφή είναι σαν την συνάρτηση y = a + bx όπου a είναι το intercept και το b είναι το slope

    ax = fig.add_subplot()
    ax.scatter(X_train[:, [0]], X_train[:, [1]], marker='o', c=predicted)
    ax.set_xlim([min(X_train[:, [0]]) - 2, max(X_train[:, [0]]) + 2])
    ax.set_ylim([min(X_train[:, [1]]) - 2, max(X_train[:, [1]] + 2)])
    ax.plot(x, y, 'r')
    plt.title("Season " + str(seasons))
    plt.draw()


if n_inputs == 3:
    z = x + y
    plt.clf()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X_train[:, [0]], X_train[:, [1]], X_train[:, [2]], marker='o', c=predicted)
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm)
    plt.title("Season-less (No Weights)")
    plt.draw()
    plt.pause(3)
    plt.clf()

    if len(Show_Seasons) >= 0:
        z = (-plotWeights[0][0] / plotWeights[0][2]) * x + (-plotWeights[0][1] / plotWeights[0][2]) * y + (
                    -plotWeights[0][n_inputs] / plotWeights[0][2])

        ax = fig.add_subplot(projection='3d')
        ax.scatter(X_train[:, [0]], X_train[:, [1]], X_train[:, [2]], marker='o', c=predicted)
        surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm)
        plt.title("Season " + str(Show_Seasons[0]))
        plt.draw()
        plt.pause(3)
        plt.clf()

    if len(Show_Seasons) >= 2:
        z = (-plotWeights[2][0] / plotWeights[2][2]) * x + (-plotWeights[2][1] / plotWeights[2][2]) * y + (
                    -plotWeights[2][n_inputs] / plotWeights[2][2])
        ax = fig.add_subplot(projection='3d')
        ax.scatter(X_train[:, [0]], X_train[:, [1]], X_train[:, [2]], marker='o', c=predicted)
        surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm)
        plt.title("Season " + str(Show_Seasons[2]))
        plt.draw()
        plt.pause(5)
        plt.clf()

    if len(Show_Seasons) >= 3:
        z = (-plotWeights[3][0] / plotWeights[3][2]) * x + (-plotWeights[3][1] / plotWeights[3][2]) * y + (
                    -plotWeights[3][n_inputs] / plotWeights[3][2])
        ax = fig.add_subplot(projection='3d')
        ax.scatter(X_train[:, [0]], X_train[:, [1]], X_train[:, [2]], marker='o', c=predicted)
        surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm)
        plt.title("Season " + str(Show_Seasons[3]))
        plt.draw()
        plt.pause(5)
        plt.clf()

if n_inputs == 3:
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X_train[:, [0]], X_train[:, [1]], X_train[:, [2]], marker='o', c=predicted)

    z = (-weights[0] / weights[2]) * x + (-weights[1] / weights[2]) * y + (-weights[n_inputs] / weights[2])
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm)
    ax.set(xlabel='X1', ylabel='X2', zlabel='X3')
    plt.title("Season " + str(seasons))
    plt.draw()
