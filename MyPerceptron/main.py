import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import random
#Copyright (c) 2022, Chilled Ferrum All rights reserved.

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

# Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

# Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

# All advertising materials mentioning features or use of this software must display the following acknowledgement: This product includes software developed by the ChilledFerrum.

# Neither the name of the ChilledFerrum nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY ChilledFerrum AS IS 
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
# IN NO EVENT SHALL ChilledFerrum BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, 
# OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


# # Parameters...
# Play around with the Parameters Here is some hints

# urlDataset = 'dataset1.csv'
# learning_rate = 0.08
# n_rows = 100
# n_inputs = 2
# act_outputs = [1, -1]

urlDataset = 'dataset3D.csv'  # Dataset URL
learning_rate = 0.05
n_rows = 100
n_inputs = 3  # Changes with the dataset
seasons = 30


weights = np.zeros(n_inputs)
bias = random.uniform(-2.0, 2.0)

# Randomize weights...
for i in range(0, n_inputs):
    weights[i] = random.uniform(-2.0, 2.0)
    if weights[i] == 0:
        weights[i] += 0.1

# Activation Function Outputs [First , Second]. 
# Change act_outputs according to the dataset's target label 
# e.g. act_outputs = ['C1', 'C2'] OR act_outputs = [1, -1]

act_outputs = ['C1', 'C2']

# Set Target Label Name...
t_label = 'Y'
# Set Label Starting Character
input_label = 'X'


def initDataset(UrlDataset, n):
    data = pd.read_csv(UrlDataset, nrows=n)
    return data


# Returns 0 if the function of the linear output is greater than 0
def activationFunc(Y):
    return 1.0 if Y > 0 else 0.0


# Load Data using pandas to numpy
dataset = initDataset(urlDataset, n_rows)
df_target = pd.DataFrame(dataset, columns=[t_label])
target = df_target.to_numpy()

# Convert Labels to 1 & 0
if n_inputs == 2:
    for idx, x in enumerate(target):
        if x == act_outputs[0]:
            target[idx] = 1.0
        elif x == act_outputs[1]:
            target[idx] = 0.0


# Convert String Labels to 1 & 0
if isinstance(act_outputs[0], str):
    target = np.zeros(n_rows)
    for col, row in df_target.iterrows():
        if row[t_label] == act_outputs[0]:
            target[col] = 1.0
        elif row[t_label] == act_outputs[1]:
            target[col] = 0.0


# Data Retrieval using pandas
Iter = 0
dfX_train = pd.DataFrame()
for col in dataset:
    if col != t_label:
        Iter += 1
        label = input_label + str(Iter)
        dfX_train[Iter] = pd.DataFrame(dataset, columns=[label])

X_train = dfX_train.to_numpy()

# plot planning/preparation for 2D dataset
if n_inputs == 2:
    x = np.arange(0, np.max(X_train[:, [0]]), 0.1)
    y = -(weights[0] / weights[1]) * x + (-bias / weights[1])

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlim([min(X_train[:, [0]]) - 2, max(X_train[:, [0]]) + 2])
    ax.set_ylim([min(X_train[:, [1]]) - 2, max(X_train[:, [1]] + 2)])

    line, = ax.plot(x, y)

# plot planning/preparation for 3D dataset
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

# Training the Dataset...
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

        # plotting 2D dataset with regression line in real time
        if n_inputs == 2:
            y = -(weights[0] / weights[1]) * x + \
                (-bias / weights[1])
            ax.set(xlabel='X1',
                   ylabel='X2',
                   title="Season: " + str(current_season) + " Iter: " + str(current_iter)
                   )
            ax.set_xlim([min(X_train[:, [0]]) - 2, max(X_train[:, [0]]) + 2])
            ax.set_ylim([min(X_train[:, [1]]) - 2, max(X_train[:, [1]] + 2)])
            ax.scatter(X_train[:, [0]], X_train[:, [1]], marker='o', c=currently_predicted)
            line, = ax.plot(x, y)
            line.set_ydata(y)

            fig.canvas.draw()
            plt.pause(0.0005)
            fig.canvas.flush_events()
            ax.clear()

        # plotting 3D dataset with regression line in real time
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
    # Loop breaks if all predictions are successful
    if miss == 0:
        break
    current_season += 1

# Finally visualize 2D plot with trained weights
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

# Finally visualize 3D plot with trained weights
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



# Prints the final weights after training both 2D & 3D
if n_inputs == 2:
    print('bias =\t', bias, "\nW1 =\t\t", weights[0], "\nW2 =\t\t", weights[1])
    print("Linear Function Formula -> Linear_Output = bias + W1*x1 + W2*x2\n", )
if n_inputs == 3:
    print('bias =\t', bias, "\nW0 = \t\t", weights[0], "\nW1 = \t\t", weights[1], "\nW2 = \t\t",
          weights[2])
    print("Linear Function Formula -> Linear_Output = bias + W1*x1 + W2*x2 + W3*x3\n")

final_target = ["" for x in range(n_rows)]
final_prediction = ["" for x in range(n_rows)]

# Conversion to the Original Labels from 1 & 0 to act_outputs
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

# finally prints the latest predictions made compared with the target as well as the success percentage 
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
