# import numpy as np
# from numpy.random import multivariate_normal
# import matplotlib.pyplot as plt
# # from scipy.stats import st
# from mpl_toolkits.mplot3d import Axes3D
#
# mu_1 = np.array([0, 0, 0])
# sigma_1 = np.array([[1, 0.3, -0.2], [0.3, 2, 0.5], [-0.2, 0.5, 1]])
# pdf_1 = lambda x: multivariate_normal(mu_1, sigma_1).pdf(x)
#
# mu_2 = np.array([-2, 1, 2])
# sigma_2 = np.array([[2, 0.4, 0], [0.4, 2, -0.7], [0, -0.7, 1]])
# pdf_2 = lambda x: multivariate_normal(mu_2, sigma_2).pdf(x)
#
# mu_3a = np.array([-1, 1, -1])
# sigma_3a = np.array([[1, 0.5, 0.3], [0.5, 1, -0.4], [0.3, -0.4, 3]])
# mu_3b = np.array([1, -2, 2])
# sigma_3b = np.array([[3, -0.5, 0.25], [-0.5, 1, 0.4], [0.25, 0.4, 1]])
# pdf_3 = lambda x: 0.5 * multivariate_normal(mu_3a, sigma_3a).pdf(x) + 0.5 * multivariate_normal(mu_3b, sigma_3b).pdf(x)
#
# priors = np.array([0.3, 0.3, 0.4])
#
# N = 10000
#
# labels = np.random.choice(3, size=N, p=priors)
#
# dataset = np.zeros((3, N))
# for i in range(N):
#     if labels[i] == 0:
#         dataset[:, i] = multivariate_normal(mu_1, sigma_1)
#     elif labels[i] == 1:
#         dataset[:, i] = multivariate_normal(mu_2, sigma_2)
#     elif labels[i] == 2:
#         if np.random.rand() < 0.5:
#             dataset[:, i] = multivariate_normal(mu_3a, sigma_3a)
#         else:
#             dataset[:, i] = multivariate_normal(mu_3b, sigma_3b)
#         pass
#
#
#
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# ax.tick_params(labelsize=8)
# X = dataset[0, labels==1]
# Y = dataset[1, labels==1]
# Z = dataset[2, labels==1]
# ax.scatter(X, Y, Z, c='b', marker='o', alpha = 0.25)
#
# X = dataset[0, labels==2]
# Y = dataset[1, labels==2]
# Z = dataset[2, labels==2]
# ax.scatter(X, Y, Z, c='y', marker='o', alpha = 0.25)
#
# X = dataset[0, labels==3]
# Y = dataset[1, labels==3]
# Z = dataset[2, labels==3]
# ax.scatter(X, Y, Z, c='r', marker='o', alpha = 0.25)
#
# ax.set_xlabel('X', fontsize=8)
# ax.set_ylabel('Y', fontsize=8)
# ax.set_zlabel('Z', fontsize=8)
# ax.set_title('Dataset visualization', fontsize=8, fontweight='bold')
# ax.legend(['Class 1', 'Class 2', 'Class 3'], fontsize=8)
#
# plt.show()
#
#
#
# ####################################################################3
#
# # Start to classify
# decisions = np.zeros(N)
# R_star = 0
#
# for k in range(N):
#     X = dataset[:, k]
#     Px = pdf_1(X)*priors[0] + pdf_2(X)*priors[1] + pdf_3(X)*priors[2]
#     p1 = pdf_1(X)*priors[0]
#     p2 = pdf_2(X)*priors[1]
#     p3 = pdf_3(X)*priors[2]
#     ind = np.argmax([p1, p2, p3])
#     maxp = [p1, p2, p3][ind]
#     # R_star = R_star + (1 - maxp/Px)
#     R_star = R_star + (Px - maxp)
#     if ind == 0:
#         decisions[k] = 1
#     elif ind == 1:
#         decisions[k] = 2
#     elif ind == 2:
#         decisions[k] = 3
#     else:
#         print("Decision label error, label number out of bounds.")








##

import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_classification


#1.Generating 10000 samples:
# Set up the means, covariances, and weights for the mixture of Gaussians
# set random seed for reproducibility
np.random.seed(0)

# set class priors
priors = [0.3, 0.3, 0.4]

# set means and covariances for class-conditional pdfs
mean_1 = np.array([0, 0, 0])
mean_2 = np.array([6, 0, 0])
mean_3 = np.array([3, 3, 3])
mean_4 = np.array([-3, 0, 0])

std_dev = 1
cov_1 = np.eye(3) * std_dev
cov_2 = np.eye(3) * std_dev
cov_3 = np.eye(3) * std_dev
cov_4 = np.eye(3) * std_dev

# define class-conditional pdfs
pdf_1 = multivariate_normal(mean_1, cov_1)
pdf_2 = multivariate_normal(mean_2, cov_2)
pdf_3 = multivariate_normal(mean_3, cov_3)
pdf_4 = multivariate_normal(mean_4, cov_4)

num_samples = 10000#10k

# define function to generate data from the mixture of Gaussians
def generate_data(num_samples):
    X = np.zeros((num_samples, 3))
    y = np.zeros(num_samples, dtype=int)
    for i in range(num_samples):
        # choose class based on class priors
        c = np.random.choice([0, 1, 2], p=priors)
        # generate sample from corresponding class-conditional pdf
        if c == 0:
            X[i, :] = pdf_1.rvs()
            y[i] = 1
        elif c == 1:
            X[i, :] = pdf_2.rvs()
            y[i] = 2
        else:
            if np.random.rand() < 0.5:
                X[i, :] = pdf_3.rvs()
            else:
                X[i, :] = pdf_4.rvs()
            y[i] = 3
    return X, y

# generate some data
X, y = generate_data(num_samples)




# plot the data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[y==1, 0], X[y==1, 1], X[y==1, 2], c='r', label='class 1')
ax.scatter(X[y==2, 0], X[y==2, 1], X[y==2, 2], c='g', label='class 2')
ax.scatter(X[y==3, 0], X[y==3, 1], X[y==3, 2], c='b', label='class 3')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.show()

def partA():
    #2.Implementing the decision rule and calculating the confusion matrix:
    # Define the decision rule that minimizes 0-1 loss
    def decide(x):
        # Compute the likelihoods under each class-conditional distribution
        likelihoods = [0, 0, 0]
        likelihoods[0] = 0.3 * multivariate_normal.pdf(x, mean_1, cov_1)
        likelihoods[1] = 0.3 * multivariate_normal.pdf(x, mean_2, cov_2)
        likelihoods[2] = 0.2 * multivariate_normal.pdf(x, mean_3, cov_3) + 0.2 * multivariate_normal.pdf(x, mean_4, cov_4)

        # Return the class with the maximum likelihood
        return np.argmax(likelihoods) + 1


    # Classify the samples using the decision rule
    predicted_labels = np.zeros(num_samples)
    for i in range(num_samples):
        predicted_labels[i] = decide(X[i, :])

    # Compute the confusion matrix
    confusion_matrix = np.zeros((3, 3))

    pdf_1 = multivariate_normal(mean=mean_1, cov=cov_1).pdf
    pdf_2 = multivariate_normal(mean=mean_2, cov=cov_2).pdf
    pdf_3 = multivariate_normal(mean=mean_3, cov=cov_3).pdf

    p_x_given_l = lambda x, l: pdf_1(x) if l == 1 else (pdf_2(x) if l == 2 else pdf_3(x))

    for i in range(num_samples):
        x = X[i, :]
        true_label = y[i]
        predicted_label = np.argmax([priors[j - 1] * p_x_given_l(x, j) for j in range(1, 4)]) + 1

        if predicted_label not in [1, 2, 3]:
            continue  # skip this sample if predicted label is out of range

        confusion_matrix[true_label - 1, predicted_label - 1] += 1

    # for i in range(num_samples):
    #     confusion_matrix[int(labels[i]) - 1, int(predicted_labels[i]) - 1] += 1
    # confusion_matrix /= np.sum(confusion_matrix, axis=1, keepdims=True)



    #3.Creating a scatter plot of the data with markers indicating true and predicted labels:
    # Define marker shapes for true class labels
    marker_shapes = {1: '.', 2: 'o', 3: '^'}

    # Define marker colors for correct and incorrect classification
    marker_colors = {True: 'g', False: 'r'}

    # Create the scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(num_samples):
        # Plot the sample with the appropriate marker shape and color
        true_label = int(y[i])
        predicted_label = int(predicted_labels[i])
        marker = marker_shapes[true_label]
        color = marker_colors[true_label == predicted_label]
        ax.scatter(X[i, 0], X[i, 1], X[i, 2], marker=marker, c=color, alpha = 0.25)

    # Add labels and title to the plot
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    ax.set_title('Scatter plot of data with true and predicted labels')

    # Add legend to the plot
    handles = []
    labels = []
    for label, shape in marker_shapes.items():
        for is_correct, color in marker_colors.items():
            handles.append(ax.scatter([], [], [], marker=shape, c=color))
            labels.append(f'Class {label}, {"Correct" if is_correct else "Incorrect"}')
    ax.legend(handles, labels, loc='upper left')

    # Show the plot
    plt.show()

partA()





###PARTB
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

# Generate a random dataset with 3 features and 3 classes
X, y = make_classification(n_samples=10000, n_features=3, n_classes=3, n_clusters_per_class=1, random_state=42)

# Define the loss matrices
Lambda10 = np.array([[0, 1, 10], [1, 0, 10], [1, 1, 0]])
Lambda100 = np.array([[0, 1, 100], [1, 0, 100], [1, 1, 0]])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a decision tree classifier on the training data
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict the class labels for the testing data
y_pred = clf.predict(X_test)

# Calculate the confusion matrix for the 0-1 loss
cm_01 = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
P_D_given_L_01 = cm_01 / np.sum(cm_01, axis=0, keepdims=True)
P_D_given_L_01 = np.nan_to_num(P_D_given_L_01)

# Calculate the confusion matrix for the Lambda10 loss
cm_L10 = confusion_matrix(y_test, y_pred, labels=[0, 1, 2], sample_weight=np.array([Lambda10[y_test[i], j] for i, j in enumerate(y_pred)]))
P_D_given_L_L10 = cm_L10 / np.sum(cm_L10, axis=0, keepdims=True)
P_D_given_L_L10 = np.nan_to_num(P_D_given_L_L10)

# Calculate the confusion matrix for the Lambda100 loss
cm_L100 = confusion_matrix(y_test, y_pred, labels=[0, 1, 2], sample_weight=np.array([Lambda100[y_test[i], j] for i, j in enumerate(y_pred)]))
P_D_given_L_L100 = cm_L100 / np.sum(cm_L100, axis=0, keepdims=True)
P_D_given_L_L100 = np.nan_to_num(P_D_given_L_L100)

# Estimate the minimum expected risk for the Lambda10 loss
risk_L10 = np.sum(np.multiply(P_D_given_L_L10, Lambda10))
print("Minimum expected risk for Lambda10:", risk_L10)

# Estimate the minimum expected risk for the Lambda100 loss
risk_L100 = np.sum(np.multiply(P_D_given_L_L100, Lambda100))
print("Minimum expected risk for Lambda100:", risk_L100)

