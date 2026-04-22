import numpy as np
from sklearn.svm import SVC

X = np.array([[6,5],[6,9],[8,6],[8,8],[8,10],[9,2],[9,5],[10,10],
              [10,13],[11,5],[11,8],[12,6],[12,11],[13,4],[14,8]])

y = np.array([0,0,1,1,1,0,1,1,0,1,1,1,0,0,0])  #Here, Blue=0, Red=1

rbf_model = SVC(kernel='rbf', gamma=0.5)
poly_model = SVC(kernel='poly', degree=3)

rbf_model.fit(X,y)
poly_model.fit(X,y)

print("RBF accuracy:", rbf_model.score(X,y))
print("Polynomial accuracy:", poly_model.score(X,y))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC


def plot(model, X, y, title):
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y)

    ax = plt.gca()
    """gca stand for get current axis. 
    xlim and ylim define plot boundaries.
    It is used to create a grid covering entire plot"""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    """ to create mesh grid"""
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    """This creates a 30x30 grid points. it gives many points across plane.
    classification is tested on each point"""

    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    """Converts grid into list of coordinate pairs"""

    """This gets the decision values"""
    Z = model.decision_function(xy).reshape(XX.shape)
    """Model predicts value for each grid point. Reshape converts back to grid form"""

    ax.contour(XX, YY, Z, levels=[0])
    """This draws the decision boundary.
    levels = [0] means draw line where prediction = 0
    This is decision boundary line. It separates Red vs Blue"""

    plt.title(title)
    plt.show()

def main():
    X = np.array([[6, 5], [6, 9], [8, 6], [8, 8], [8, 10], [9, 2], [9, 5], [10, 10],
                  [10, 13], [11, 5], [11, 8], [12, 6], [12, 11], [13, 4], [14, 8]])

    y = np.array([0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0])  # Here, Blue=0, Red=1

    # RBF Kernel
    rbf_model = SVC(kernel='rbf', gamma=0.5)
    rbf_model.fit(X, y)

    # Polynomial Kernel
    poly_model = SVC(kernel='poly', degree=3)
    poly_model.fit(X, y)

    # Accuracy
    print("RBF Accuracy:", rbf_model.score(X, y))
    print("Polynomial Accuracy:", poly_model.score(X, y))

    # Plot
    plot(rbf_model, X, y, "RBF Kernel")
    plot(poly_model, X, y, "Polynomial Kernel")


if __name__ == "__main__":
    main()