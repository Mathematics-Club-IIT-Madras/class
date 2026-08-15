
"""
Logistic Regression Dataset Generator

Distributions:
1. Normal0   : X ~ N(0, Σ)
2. Normal1   : X ~ N(1, Σ)
3. MixNormal : 50% N(0, Σ) + 50% N(1, Σ)
4. T3        : X ~ t_3(0, Σ/10)
5. All

Default:
    Covariates = 250
    Active     = 50

Outputs:
    dataset CSV
    metadata TXT
"""

import numpy as np
import pandas as pd
from scipy.stats import multivariate_t
from datetime import datetime


# ==========================================================
# Utility Functions
# ==========================================================

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def generate_beta(p, active, rng):
    beta = np.zeros(p)

    active_idx = rng.choice(
        p,
        size=active,
        replace=False
    )

    beta[active_idx] = rng.normal(
        loc=0.0,
        scale=1.0,
        size=active
    )

    return beta, np.sort(active_idx)


def generate_labels(X, beta, rng):

    logits = X @ beta

    probs = sigmoid(logits)

    y = rng.binomial(
        n=1,
        p=probs
    )

    return y


# ==========================================================
# Covariance Matrix
# ==========================================================

def build_covariance(p, choice):

    if choice == "1":

        cov_name = "Identity"

        Sigma = np.eye(p)

    elif choice == "2":

        cov_name = "AR(1), rho = 0.5"

        rho = 0.5

        Sigma = np.fromfunction(
            lambda i, j: rho ** np.abs(i - j),
            (p, p)
        )

    else:
        raise ValueError(
            "Invalid covariance choice."
        )

    return Sigma, cov_name


# ==========================================================
# Dataset Generators
# ==========================================================

def generate_normal0(n, Sigma):

    p = Sigma.shape[0]

    return np.random.multivariate_normal(
        mean=np.zeros(p),
        cov=Sigma,
        size=n
    )


def generate_normal1(n, Sigma):

    p = Sigma.shape[0]

    return np.random.multivariate_normal(
        mean=np.ones(p),
        cov=Sigma,
        size=n
    )


def generate_mixnormal(n, Sigma):

    p = Sigma.shape[0]

    n0 = n // 2
    n1 = n - n0

    X0 = np.random.multivariate_normal(
        mean=np.zeros(p),
        cov=Sigma,
        size=n0
    )

    X1 = np.random.multivariate_normal(
        mean=np.ones(p),
        cov=Sigma,
        size=n1
    )

    X = np.vstack([X0, X1])

    np.random.shuffle(X)

    return X


def generate_t3(n, Sigma):

    p = Sigma.shape[0]

    return multivariate_t.rvs(
        loc=np.zeros(p),
        shape=Sigma / 10.0,
        df=3,
        size=n
    )


# ==========================================================
# Metadata
# ==========================================================

def write_metadata(
        filename,
        dataset_name,
        n,
        p,
        active,
        beta,
        active_idx,
        cov_name,
        Sigma,
        class_balance,
        seed):

    with open(filename, "w") as f:

        f.write("=" * 70 + "\n")
        f.write("DATASET METADATA\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Dataset Type          : {dataset_name}\n")
        f.write(f"Rows                  : {n}\n")
        f.write(f"Covariates            : {p}\n")
        f.write(f"Active Covariates     : {active}\n")
        f.write(f"Random Seed           : {seed}\n")
        f.write(f"Covariance Structure  : {cov_name}\n")
        f.write(f"Positive Class Ratio  : {class_balance:.4f}\n")
        f.write(f"Generated             : {datetime.now()}\n")

        f.write("\n")
        f.write("=" * 70 + "\n")
        f.write("ACTIVE FEATURE INDICES\n")
        f.write("=" * 70 + "\n\n")

        f.write(str(active_idx.tolist()))

        f.write("\n\n")
        f.write("=" * 70 + "\n")
        f.write("TRUE BETA VECTOR\n")
        f.write("=" * 70 + "\n\n")

        np.set_printoptions(
            precision=4,
            suppress=True
        )

        f.write(np.array2string(beta))

        if p <= 20:

            f.write("\n\n")
            f.write("=" * 70 + "\n")
            f.write("COVARIANCE MATRIX\n")
            f.write("=" * 70 + "\n\n")

            f.write(np.array2string(
                Sigma,
                precision=3
            ))

        else:

            f.write("\n\n")
            f.write(
                "Covariance matrix omitted "
                "(p > 20).\n"
            )


# ==========================================================
# Save Dataset
# ==========================================================

def save_dataset(
        dataset_name,
        X,
        beta,
        active_idx,
        Sigma,
        cov_name,
        seed,
        rng):

    y = generate_labels(
        X,
        beta,
        rng
    )

    columns = [
        f"x{i+1}"
        for i in range(X.shape[1])
    ]

    df = pd.DataFrame(
        X,
        columns=columns
    )

    df["y"] = y

    csv_name = (
        f"{dataset_name}"
        f"_n{X.shape[0]}"
        f"_p{X.shape[1]}.csv"
    )

    meta_name = (
        f"{dataset_name}"
        f"_metadata.txt"
    )

    df.to_csv(
        csv_name,
        index=False
    )

    write_metadata(
        filename=meta_name,
        dataset_name=dataset_name,
        n=X.shape[0],
        p=X.shape[1],
        active=len(active_idx),
        beta=beta,
        active_idx=active_idx,
        cov_name=cov_name,
        Sigma=Sigma,
        class_balance=y.mean(),
        seed=seed
    )

    print(f"Saved: {csv_name}")
    print(f"Saved: {meta_name}")
    print()


# ==========================================================
# Main
# ==========================================================

def main():

    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION DATASET GENERATOR")
    print("=" * 60)

    n = input(
        "\nNumber of rows [10000]: "
    ).strip()

    p = input(
        "Number of covariates [250]: "
    ).strip()

    active = input(
        "Number of active covariates [50]: "
    ).strip()

    seed = input(
        "Random seed [42]: "
    ).strip()

    print("\nChoose Distribution")
    print("-------------------")
    print("1. Normal0")
    print("2. Normal1")
    print("3. MixNormal")
    print("4. T3")
    print("5. All")

    dataset_choice = input(
        "\nChoice [5]: "
    ).strip()

    print("\nChoose Covariance Structure")
    print("---------------------------")
    print("1. Identity")
    print("2. AR(1)")

    cov_choice = input(
        "\nChoice [1]: "
    ).strip()

    n = int(n) if n else 10000
    p = int(p) if p else 250
    active = int(active) if active else 50
    seed = int(seed) if seed else 42

    dataset_choice = (
        dataset_choice
        if dataset_choice
        else "5"
    )

    cov_choice = (
        cov_choice
        if cov_choice
        else "1"
    )

    if active > p:

        raise ValueError(
            "Active covariates "
            "cannot exceed total covariates."
        )

    rng = np.random.default_rng(seed)

    np.random.seed(seed)

    Sigma, cov_name = build_covariance(
        p,
        cov_choice
    )

    beta, active_idx = generate_beta(
        p,
        active,
        rng
    )

    generators = {
        "1": ("normal0", generate_normal0),
        "2": ("normal1", generate_normal1),
        "3": ("mixnormal", generate_mixnormal),
        "4": ("t3", generate_t3)
    }

    if dataset_choice == "5":

        for name, generator in generators.values():

            X = generator(
                n,
                Sigma
            )

            save_dataset(
                name,
                X,
                beta,
                active_idx,
                Sigma,
                cov_name,
                seed,
                rng
            )

    else:

        if dataset_choice not in generators:

            raise ValueError(
                "Invalid dataset choice."
            )

        name, generator = generators[
            dataset_choice
        ]

        X = generator(
            n,
            Sigma
        )

        save_dataset(
            name,
            X,
            beta,
            active_idx,
            Sigma,
            cov_name,
            seed,
            rng
        )

    print("=" * 60)
    print("Generation Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
