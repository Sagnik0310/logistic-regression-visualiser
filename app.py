import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Load initial graph
def load_initial_graph(dataset, ax):
    if dataset == "Binary":
        X, y = make_blobs(
            n_features=2,
            centers=2,
            random_state=6
        )
    else:
        X, y = make_blobs(
            n_features=2,
            centers=3,
            random_state=6
        )

    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="rainbow")
    return X, y


# Draw meshgrid
def draw_meshgrid(X):
    a = np.arange(X[:, 0].min() - 1,
                  X[:, 0].max() + 1,
                  0.01)

    b = np.arange(X[:, 1].min() - 1,
                  X[:, 1].max() + 1,
                  0.01)

    XX, YY = np.meshgrid(a, b)

    input_array = np.c_[XX.ravel(), YY.ravel()]

    return XX, YY, input_array


plt.style.use("fivethirtyeight")

st.sidebar.title("Logistic Regression Visualiser")

dataset = st.sidebar.selectbox(
    "Select Dataset",
    ["Binary", "Multi-class"]
)

penalty = st.sidebar.selectbox(
    "Regularization",
    ["l1", "l2", "elasticnet", "none"]
)

c_input = float(
    st.sidebar.number_input(
        "C",
        value=1.0
    )
)

solver = st.sidebar.selectbox(
    "Solver",
    ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
)

max_iter = int(
    st.sidebar.number_input(
        "Max Iterations",
        value=100
    )
)

# Only for display (not used in sklearn 1.9)
st.sidebar.selectbox(
    "Multi-class",
    ["auto", "ovr", "multinomial"]
)

l1_ratio = float(
    st.sidebar.number_input(
        "l1_ratio",
        value=0.5
    )
)


fig, ax = plt.subplots()

X, y = load_initial_graph(dataset, ax)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

orig = st.pyplot(fig)


if st.sidebar.button("Run the algorithm"):

    orig.empty()

    # Convert "none" to None
    penalty_used = None if penalty == "none" else penalty

    # Elastic Net requires saga
    if penalty_used == "elasticnet" and solver != "saga":
        st.error("Elastic Net only works with solver='saga'")
        st.stop()

    # L1 compatibility
    if penalty_used == "l1" and solver not in ["liblinear", "saga"]:
        st.error("L1 penalty only works with 'liblinear' or 'saga'")
        st.stop()

    # None compatibility
    if penalty_used is None and solver == "liblinear":
        st.error("No regularization is not supported with liblinear")
        st.stop()

    # Build model
    if penalty_used == "elasticnet":

        clf = LogisticRegression(
            penalty=penalty_used,
            C=c_input,
            solver=solver,
            max_iter=max_iter,
            l1_ratio=l1_ratio
        )

    else:

        clf = LogisticRegression(
            penalty=penalty_used,
            C=c_input,
            solver=solver,
            max_iter=max_iter
        )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    XX, YY, input_array = draw_meshgrid(X)

    labels = clf.predict(input_array)

    labels = labels.reshape(XX.shape)

    ax.clear()

    ax.contourf(
        XX,
        YY,
        labels,
        alpha=0.5,
        cmap="rainbow"
    )

    ax.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap="rainbow",
        edgecolors="black"
    )

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("Decision Boundary")

    st.pyplot(fig)

    st.subheader(
        f"Accuracy for Logistic Regression: {accuracy_score(y_test, y_pred):.2f}"
    )