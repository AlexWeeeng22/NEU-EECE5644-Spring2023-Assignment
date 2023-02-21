# Data_Scale = 100000 #100,000
#
# def gen_clusters():
#     mean1 = [-0.5, -0.5, -0.5, -0.5]
#     cov1 = [[0.5, -0.125, 0.075, 0], [-0.125, 0.25, -0.125, 0], [0.075, -0.125, 0.25, 0], [0, 0, 0, 0.5]]
#     data = np.random.multivariate_normal(mean1, cov1, Data_Scale)
#
#     mean2 = [1, 1, 1, 1]
#     cov2 = [[1, 0.3, -0.2, 0], [0.3, 2, 0.3, 0], [-0.2, 0.3, 1, 0], [0, 0, 0, 3]]
#     data = np.append(data,
#                      np.random.multivariate_normal(mean2, cov2, Data_Scale),
#                      0)
#     return np.round(data, 9)
#
#
# def save_data(data, filename):
#     with open(filename, 'w') as file:
#         for i in range(data.shape[0]):
#             file.write(str(data[i, 0]) + ',' + str(data[i, 1]) + '\n')
#
#
# def load_data(filename):
#     data = []
#     with open(filename, 'r') as file:
#         for line in file.readlines():
#             data.append([float(i) for i in line.split(',')])
#     return np.array(data)
#
#
# def show_scatter(data):
#     x, y = data.T
#     plt.scatter(x, y)
#     plt.axis()
#     plt.title("scatter")
#     plt.xlabel("x")
#     plt.ylabel("y")
#     plt.show()
#
#
# data = gen_clusters()
# save_data(data, '3clusters.txt')
# d = load_data('3clusters.txt')
# show_scatter(d)

#////////////////////////////////////////////////////////////////////////////////////////////////////////////

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

#Original Parameters
m_0 = np.array([-1/2, -1/2, -1/2, -1/2]).reshape(-1, 1)
C_0 = np.array([[0.5, -0.125, 0.075, 0], [-0.125, 0.25, -0.125, 0], [0.075, -0.125, 0.25, 0], [0, 0, 0, 0.5]])
m_1 = np.array([1, 1, 1, 1]).reshape(-1, 1)
C_1 = np.array([[1, 0.3, -0.2, 0], [0.3, 2, 0.3, 0], [-0.2, 0.3, 1, 0], [0, 0, 0, 3]])

priors = np.array([0.35, 0.65])
N = int(1e4)#data scale 10K

labels_raw = np.random.choice([0, 1], size = N, p = priors)

def generate_data(labels_raw, m_0, C_0, m_1, C_1):
    data = np.array([label * np.random.multivariate_normal(m_1.flatten(), C_1) + (1 - label) * np.random.multivariate_normal(m_0.flatten(), C_0) for label in labels_raw])
    return data

def show_scatter_2D(data):
    color = ['r' if label == 0 else 'b' for label in labels_raw]
    x, y = data[:, :2].T
    plt.scatter(x, y, c = color, alpha = 0.5)
    plt.title("Dataset Scatter Plot")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def show_scatter_3D(data, labels_raw):
    color = ['r' if label == 0 else 'b' for label in labels_raw]
    fig = plt.figure()# data is a 2D array with shape (N, 3)
    ax = Axes3D(fig)

    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c = color, alpha = 0.5) # Plot the data as a scatter plot

    ax.set_xlabel('x') # Set labels for the axes
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()

#show generated 100000 datapoint in 2D and 3D view
def show_raw_data_graph():
    dataset = generate_data(labels_raw, m_0, C_0, m_1, C_1)
    show_scatter_2D(dataset)
    show_scatter_3D(dataset, labels_raw)


#PART A
#////////////////////////////////////////////////////////////////////////////////////////////////////////////

def part_A():
    labels_logic = labels_raw.astype(bool)

    steps = 200

    tp = np.zeros(steps)  # true positive
    fp = np.zeros(steps)  # false positive
    tn = np.zeros(steps)  # true negative
    fn = np.zeros(steps)  # false negative

    def generate_gamma():
        gamma = np.concatenate((np.linspace(0, 4, steps//2, endpoint=False),
                                np.exp(np.linspace(np.log(6), 20, steps//2)) - 1))
        return gamma
    gamma = generate_gamma()

    #Overall, this code appears to be performing threshold optimization
    #for the binary classifier using a Bayesian approach.
    # By adjusting the threshold values gamma and calculating the TPR and FPR for each threshold,
    # the optimal threshold can be chosen based on the specific needs of the classification problem.
    for k in range(steps):
        decisions = np.array([multivariate_normal.pdf(X, m_1.flatten(), C_1) / multivariate_normal.pdf(X, m_0.flatten(), C_0) > gamma[k] for X in dataset])
        tp[k] = sum(labels_logic & decisions)
        fp[k] = sum(~labels_logic & decisions)
        tn[k] = sum(~labels_logic & ~decisions)
        fn[k] = sum(labels_logic & ~decisions)

    tpr = tp / sum(labels_logic)  # True Positive Rate (TPR)
    fpr = fp / (N - sum(labels_logic))  # False Positive Rate (FPR)

    #Minimum P(error) calculation
    error_exp = (fp + fn) / N
    ind_min_err = np.argmin(error_exp)
    min_error_exp = error_exp[ind_min_err]
    min_err_gamma = gamma[ind_min_err]

    # Decision rule and TPR, FPR, P(error) calculations for 'theoretically optimal' threshold
    decisions_opt = np.array([multivariate_normal.pdf(X, mean=m_1.flatten(), cov=C_1) / multivariate_normal.pdf(X, mean=m_0.flatten(), cov=C_0) > priors[0]/priors[1] for X in dataset])
    tpr_opt = sum(labels_logic & decisions_opt) / sum(labels_logic)
    fpr_opt = sum(~labels_logic & decisions_opt) / (N - sum(labels_logic))
    err_opt = (sum(~labels_logic & decisions_opt) + sum(labels_logic & ~decisions_opt)) / N

    print(f'Empirical min. error: {min_error_exp} | Theoretical min. error {err_opt:.4f}')
    print(f'Empirical (TPR,FPR): ({tpr[ind_min_err]}, {fpr[ind_min_err]:.4f}) | Theoretical (TPR,FRP): ({tpr_opt:.4f}, {fpr_opt:.4f})')


    plt.figure()
    plt.plot(fpr, tpr, linewidth=1.5)
    plt.plot(fpr[ind_min_err], tpr[ind_min_err], 'r*', linewidth=1.0)
    plt.ylabel('TPR', fontsize=8)
    plt.xlabel('FPR', fontsize=8)
    plt.title('ROC Curve', fontsize=8)
    plt.text(fpr[ind_min_err]+0.01, tpr[ind_min_err], f'$\\leftarrow$ Minimum error point, $\\gamma_{{emp}} = {min_err_gamma:.4f}$', fontsize=8)
    plt.text(fpr[ind_min_err]+0.01, tpr[ind_min_err]-0.1, f'$(TPR_{{emp}},FPR_{{emp}}) = ({tpr[ind_min_err]:.4f},{fpr[ind_min_err]:.4f})$', fontsize=8)
    plt.show()

    # Superimposed plot (part A)
    fig_simp = plt.figure()
    plt.plot(fpr, tpr, 'b', linewidth=1.5)
    plt.ylabel('True Positive Rate', fontsize=8)
    plt.xlabel('False Positive Rate', fontsize=8)
    plt.title('ROC Curves', fontweight='bold', fontsize=8)
    ax_simp = plt.gca()
    ax_simp.tick_params(labelsize=8)

#PART B
#////////////////////////////////////////////////////////////////////////////////////////////////////////////

def part_B():
    steps = 300

    def generate_gamma():
        gamma = np.concatenate((np.linspace(0, 4, steps//2, endpoint=False),
                                np.exp(np.linspace(np.log(6), 20, steps//2)) - 1))
        return gamma
    gamma = generate_gamma()

    tp = np.zeros(steps)  # true positive
    fp = np.zeros(steps)  # false positive
    tn = np.zeros(steps)  # true negative
    fn = np.zeros(steps)  # false negative

    labels_logic = labels_raw.astype(bool)

    for k in range(steps):
        decisions = np.array([multivariate_normal.pdf(X, mean=m_1.flatten(), cov=C_1) / multivariate_normal.pdf(X, mean=m_0.flatten(), cov=C_0) > gamma[k] for X in dataset])
        tp[k] = sum(labels_logic & decisions)
        fp[k] = sum(~labels_logic & decisions)
        tn[k] = sum(~labels_logic & ~decisions)
        fn[k] = sum(labels_logic & ~decisions)

    tpr = tp / sum(labels_logic)  # True Positive Rate (TPR)
    fpr = fp / (N - sum(labels_logic))  # False Positive Rate (FPR)

    #Minimum P(error) calculation
    error_exp = (fp + fn) / N
    ind_min_err = np.argmin(error_exp)
    min_error_exp = error_exp[ind_min_err]
    min_err_gamma = gamma[ind_min_err]

    # Decision rule and TPR, FPR, P(error) calculations for 'theoretically optimal' threshold
    decisions_opt = np.array([multivariate_normal.pdf(X, mean=m_1.flatten(), cov=C_1) / multivariate_normal.pdf(X, mean=m_0.flatten(), cov=C_0) > priors[0]/priors[1] for X in dataset])
    tpr_opt = sum(labels_logic & decisions_opt) / sum(labels_logic)
    fpr_opt = sum(~labels_logic & decisions_opt) / (N - sum(labels_logic))
    err_opt = (sum(~labels_logic & decisions_opt) + sum(labels_logic & ~decisions_opt)) / N

    print(f'Empirical min. error: {min_error_exp} | Theoretical min. error {err_opt:.4f}')
    print(f'Empirical (TPR,FPR): ({tpr[ind_min_err]}, {fpr[ind_min_err]:.4f}) | Theoretical (TPR,FRP): ({tpr_opt:.4f}, {fpr_opt:.4f})')

    # Naive Bayes ROC curve plot
    plt.figure()
    plt.plot(fpr, tpr, linewidth=1.5)
    plt.xlabel('False Positive Rate', fontsize=8)
    plt.ylabel('True Positive Rate', fontsize=8)
    plt.title('ROC Curve (with incorrect $\Sigma$ assumption)', fontweight='bold', fontsize=8, pad=10, loc='left')
    plt.plot(fpr[ind_min_err], tpr[ind_min_err], 'r*', linewidth=1.5)
    plt.text(fpr[ind_min_err]+0.01, tpr[ind_min_err],
             f'$\\leftarrow$ Minimum error point $\\gamma_{{emp}}$ = {min_err_gamma:.4f}',
             fontsize=8)
    plt.text(fpr[ind_min_err]+0.01, tpr[ind_min_err]-0.1,
             f'$(TPR_{{emp}},FPR_{{emp}}) = ({tpr[ind_min_err]:.4f},{fpr[ind_min_err]:.4f})$',
             fontsize=8)

    # Superimposed plot (part B)
    plt.plot(fpr, tpr, 'r--', linewidth=1.5)
    plt.legend(['True data', 'Naive Bayes'], fontsize=8, loc='lower right')
    plt.show()

#PART C
#////////////////////////////////////////////////////////////////////////////////////////////////////////////

def part_C():
    labels_logic = labels_raw.astype(bool)

    steps = 300

    #initialize
    tp = np.zeros(steps)  # true positive
    fp = np.zeros(steps)  # false positive
    tn = np.zeros(steps)  # true negative
    fn = np.zeros(steps)  # false negative


    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    # Step 1: Generate gamma values
    gamma = np.concatenate((np.linspace(0, 4, steps//2, endpoint=False),
                            np.exp(np.linspace(np.log(6), 20, steps//2)) - 1))

    # Step 2: Fit LDA to the data
    lda = LinearDiscriminantAnalysis()
    lda.fit(dataset, labels_logic)

    # Step 3: Compute signed distances to hyperplane
    distances = lda.decision_function(dataset)

    # Step 4: Use signed distances as decision function for binary classifier
    for k in range(steps):
        decisions = (distances > gamma[k]).astype(int)
        tp[k] = sum(labels_logic & decisions)
        fp[k] = sum(~labels_logic & decisions)
        tn[k] = sum(~labels_logic & ~decisions)
        fn[k] = sum(labels_logic & ~decisions)

    tpr = tp / sum(labels_logic)  # True Positive Rate (TPR)
    fpr = fp / (N - sum(labels_logic))  # False Positive Rate (FPR)

    #Minimum P(error) calculation
    error_exp = (fp + fn) / N
    ind_min_err = np.argmin(error_exp)
    min_error_exp = error_exp[ind_min_err]
    min_err_gamma = gamma[ind_min_err]


    # def generate_gamma():
    #     gamma = np.concatenate((np.linspace(0, 4, steps//2, endpoint=False),
    #                             np.exp(np.linspace(np.log(6), 20, steps//2)) - 1))
    #     return gamma
    # gamma = generate_gamma()
    #
    # #Overall, this code appears to be performing threshold optimization
    # #for the binary classifier using a Bayesian approach.
    # # By adjusting the threshold values gamma and calculating the TPR and FPR for each threshold,
    # # the optimal threshold can be chosen based on the specific needs of the classification problem.
    # for k in range(steps):
    #     decisions = np.array([multivariate_normal.pdf(X, m_1.flatten(), C_1) / multivariate_normal.pdf(X, m_0.flatten(), C_0) > gamma[k] for X in dataset])
    #     tp[k] = sum(labels_logic & decisions)
    #     fp[k] = sum(~labels_logic & decisions)
    #     tn[k] = sum(~labels_logic & ~decisions)
    #     fn[k] = sum(labels_logic & ~decisions)
    #
    # tpr = tp / sum(labels_logic)  # True Positive Rate (TPR)
    # fpr = fp / (N - sum(labels_logic))  # False Positive Rate (FPR)
    #
    # #Minimum P(error) calculation
    # error_exp = (fp + fn) / N
    # ind_min_err = np.argmin(error_exp)
    # min_error_exp = error_exp[ind_min_err]
    # min_err_gamma = gamma[ind_min_err]

    # Decision rule and TPR, FPR, P(error) calculations for 'theoretically optimal' threshold
    decisions_opt = np.array([multivariate_normal.pdf(X, mean=m_1.flatten(), cov=C_1) / multivariate_normal.pdf(X, mean=m_0.flatten(), cov=C_0) > priors[0]/priors[1] for X in dataset])
    tpr_opt = sum(labels_logic & decisions_opt) / sum(labels_logic)
    fpr_opt = sum(~labels_logic & decisions_opt) / (N - sum(labels_logic))
    err_opt = (sum(~labels_logic & decisions_opt) + sum(labels_logic & ~decisions_opt)) / N

    print(f'Empirical min. error: {min_error_exp} | Theoretical min. error {err_opt:.4f}')
    print(f'Empirical (TPR,FPR): ({tpr[ind_min_err]}, {fpr[ind_min_err]:.4f}) | Theoretical (TPR,FRP): ({tpr_opt:.4f}, {fpr_opt:.4f})')


    plt.figure()
    plt.plot(fpr, tpr, linewidth=1.5)
    plt.plot(fpr[ind_min_err], tpr[ind_min_err], 'r*', linewidth=1.0)
    plt.ylabel('TPR', fontsize=8)
    plt.xlabel('FPR', fontsize=8)
    plt.title('ROC Curve', fontsize=8)

    plt.text(fpr[ind_min_err]+0.01, tpr[ind_min_err], f'$\\leftarrow$ Minimum error point, $\\gamma_{{emp}} = {min_err_gamma:.4f}$', fontsize=8)
    plt.text(fpr[ind_min_err]+0.01, tpr[ind_min_err]-0.1, f'$(TPR_{{emp}},FPR_{{emp}}) = ({tpr[ind_min_err]:.4f},{fpr[ind_min_err]:.4f})$', fontsize=8)
    plt.show()

    # Superimposed plot (part A)
    fig_simp = plt.figure()
    plt.plot(fpr, tpr, 'b', linewidth=1.5)
    plt.ylabel('True Positive Rate', fontsize=8)
    plt.xlabel('False Positive Rate', fontsize=8)
    plt.title('ROC Curves', fontweight='bold', fontsize=8)
    ax_simp = plt.gca()
    ax_simp.tick_params(labelsize=8)
    plt.show()

dataset = generate_data(labels_raw, m_0, C_0, m_1, C_1)
# show_raw_data_graph()
# part_A()
part_B()
#part_C()