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

urlDataset = 'dataset3D.csv'  # Dataset URL
learning_rate = 0.05
n_rows = 100
n_inputs = 3  # Changes with the dataset
seasons = 30


weights = np.zeros(n_inputs)  # +1 is for the included weight for the bias
bias = random.uniform(-2.0, 2.0)

# Randomize weights...
for i in range(0, n_inputs):
    weights[i] = random.uniform(-2.0, 2.0)
    if weights[i] == 0:
        weights[i] += 0.1

# SelectedPredictions = np.zeros(n_rows)
Show_Seasons = [1, 3, 10, 18]
plotWeights = np.zeros(shape=(len(Show_Seasons), n_inputs + 1))

# Activation Function Outputs [First , Second]. Changes according to the dataset that is being used
act_outputs = ['C1', 'C2']

# Set Target Label Name...
t_label = 'Y'
# Set Label Starting Character
input_label = 'X'


def initDataset(UrlDataset, n):
    data = pd.read_csv(UrlDataset, nrows=n)
    return data


# Returns 0 if the function of the linear output is greater or equal than 0
def activationFunc(Y):
    return 1.0 if Y > 0 else 0.0


# Load Data
dataset = initDataset(urlDataset, n_rows)
df_target = pd.DataFrame(dataset, columns=[t_label])
target = df_target.to_numpy()

if n_inputs == 2:
    for idx, x in enumerate(target):
        if x == act_outputs[0]:
            target[idx] = 1.0
        elif x == act_outputs[1]:
            target[idx] = 0.0

if isinstance(act_outputs[0], str):
    target = np.zeros(n_rows)
    for col, row in df_target.iterrows():
        if row[t_label] == act_outputs[0]:
            target[col] = 1.0
        elif row[t_label] == act_outputs[1]:
            target[col] = 0.0


Iter = 0
dfX_train = pd.DataFrame()
for col in dataset:
    if col != t_label:
        Iter += 1
        label = input_label + str(Iter)
        dfX_train[Iter] = pd.DataFrame(dataset, columns=[label])

X_train = dfX_train.to_numpy()

if n_inputs == 2:
    x = np.arange(0, np.max(X_train[:, [0]]), 0.1)
    y = -(weights[0] / weights[1]) * x + (-bias / weights[1])

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlim([min(X_train[:, [0]]) - 2, max(X_train[:, [0]]) + 2])
    ax.set_ylim([min(X_train[:, [1]]) - 2, max(X_train[:, [1]] + 2)])

    line, = ax.plot(x, y)

# Activates only if n_inputs == 3
elif n_inputs == 3:
    x = np.arange(0, np.max(X_train[:, [0]]), 0.1)
    y = np.arange(0, np.max(X_train[:, [1]]), 0.1)
    x, y = np.meshgrid(x, y)
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim([min(X_train[:, [0]]) - 2, max(X_train[:, [0]]) + 2])
    ax.set_ylim([min(X_train[:, [1]]) - 2, max(X_train[:, [1]] + 2)])
    surf = ax

predicted = np.ones(n_rows)
current_season = 0
ShowIter = 0
currently_predicted = np.zeros(n_rows)
while current_season < seasons:
    miss = 0
    for current_iter, xi in enumerate(X_train):
        linear_output = 0
        for i in range(0, n_inputs):
            linear_output += xi[i] * weights[i]

        linear_output += bias
        predicted_y = activationFunc(linear_output)
        currently_predicted[current_iter] = predicted_y
        if predicted_y != target[current_iter]:
            for i in range(0, n_inputs):
                weights[i] = weights[i] + learning_rate * (target[current_iter] - predicted_y) * xi[i]
            miss += 1

            bias = bias + learning_rate * (target[current_iter] - predicted_y)
        else:
            predicted[current_iter] = predicted_y
        if n_inputs == 2:
            y = -(weights[0] / weights[1]) * x + \
                (-bias / weights[1])
            ax.set(xlabel='X1',
                   ylabel='X2',
                   title="Season: " + str(current_season) + " Iter: " + str(current_iter)
                   )
            ax.set_xlim([min(weights[0]) - 2, max(weights[0]) + 2])
            ax.set_ylim([min(weights[1]) - 2, max(weights[1]) + 2])
            ax.scatter(X_train[:, [0]], X_train[:, [1]], marker='o', c=currently_predicted)
            line, = ax.plot(x, y)
            line.set_ydata(y)

            fig.canvas.draw()
            plt.pause(0.0005)
            fig.canvas.flush_events()
            ax.clear()

        if n_inputs == 3:
            z = (-weights[0] / weights[2]) * x + \
                (-weights[1] / weights[2]) * y + \
                (-bias / weights[2])
            plt.title("Season: " + str(current_season) + " Iter: " + str(current_iter))
            ax.set_xlim([min(X_train[:, [0]]) - 2, max(X_train[:, [0]]) + 2])
            ax.set_ylim([min(X_train[:, [1]]) - 2, max(X_train[:, [1]] + 2)])
            ax.set_zlim([min(X_train[:, [2]]) - 2, max(X_train[:, [2]] + 2)])
            ax.scatter(X_train[:, [0]], X_train[:, [1]], X_train[:, [2]], marker='o', c=currently_predicted)
            ax.scatter(xi[0], xi[1], marker='o', c='r')
            surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm)

            fig.canvas.draw()
            plt.pause(0.005)
            fig.canvas.flush_events()
            ax.clear()

    if miss == 0:
        break
    if current_season in Show_Seasons:
        for i in range(0, n_inputs):
            plotWeights[ShowIter][i] = weights[i]
        ShowIter += 1
    current_season += 1

if n_inputs == 2:
    y = -(weights[0] / weights[1]) * x + (-bias / weights[1])

    fig = plt.figure()
    ax = fig.add_subplot()
    plt.title("Season: " + str(current_season) + " Iter: " + str(current_iter))
    ax.set_xlim([min(X_train[:, [0]]) - 2, max(X_train[:, [0]]) + 2])
    ax.set_ylim([min(X_train[:, [1]]) - 2, max(X_train[:, [1]] + 2)])
    ax.scatter(X_train[:, [0]], X_train[:, [1]], marker='o', c=currently_predicted)
    ax.plot(x, y, 'r')
    plt.draw()

if n_inputs == 3:
    z = (-weights[0] / weights[2]) * x + (-weights[1] / weights[2]) * y + (-bias / weights[2])
    ax.set(xlabel='X1',
           ylabel='X2',
           zlabel='X3',
           title="Season: " + str(current_season) + " Iter: " + str(current_iter)
           )
    ax.set_xlim([min(X_train[:, [0]]) - 2, max(X_train[:, [0]]) + 2])
    ax.set_ylim([min(X_train[:, [1]]) - 2, max(X_train[:, [1]] + 2)])
    ax.set_zlim([min(X_train[:, [2]]) - 2, max(X_train[:, [2]] + 2)])

    ax.scatter(X_train[:, [0]], X_train[:, [1]], X_train[:, [2]], marker='o', c=currently_predicted)
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm)
    fig.canvas.draw()




if n_inputs == 2:
    print('bias =\t', bias, "\nW1 =\t\t", weights[0], "\nW2 =\t\t", weights[1])
    print("Linear Function Formula -> Linear_Output = W0*bias + W1*x1 + W2*x2\n", )
if n_inputs == 3:
    print('W3(bias) =\t', bias, "\nW0 = \t\t", weights[0], "\nW1 = \t\t", weights[1], "\nW2 = \t\t",
          weights[2])
    print("Linear Function Formula -> Linear_Output = W0*bias + W1*x1 + W2*x2 + W3*x3\n")

final_target = ["" for x in range(n_rows)]
final_prediction = ["" for x in range(n_rows)]

# Conversion...
for i, x in enumerate(target):
    if x == 1:
        final_target[i] = act_outputs[0]
    elif x == 0:
        final_target[i] = act_outputs[1]

for i, x in enumerate(currently_predicted):
    if x == 1:
        final_prediction[i] = act_outputs[0]
    elif x == 0:
        final_prediction[i] = act_outputs[1]

AllPredicted = True
countSuccess = 0
for i in range(0, n_rows):
    if target[i] != currently_predicted[i]:
        AllPredicted = False
    else:
        countSuccess += 1
    print("Last Predicted Values: ", final_prediction[i], "\tExpected Value: ", final_target[i])
    # print("Last Predicted Values: ", predicted[i], "\tExpected Value: ", target[i])

if AllPredicted:
    print("All Training Data Predicted at ", countSuccess / n_rows * 100, '% Accuracy')
if not AllPredicted:
    print("Not all training Data was Predicted!", countSuccess / n_rows * 100, '% Accurate')
