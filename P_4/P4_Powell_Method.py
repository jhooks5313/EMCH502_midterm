"""
Midterm P_4 - EMCH 502
Author: JP Hooks
Powell's method
Objective:  f(x1,x2) = x1 - x2 + 2*x1^2 + x2^2 + 2*x1*x2
Starting point: X = [0, 0]
"""
import numpy as np

# objective function
def f(X):
    x1, x2 = X
    return x1 - x2 + 2.0*x1**2 + x2**2 + 2.0*x1*x2

# golden-section line search  (1d minimisation)
def golden_section_search(func, X, S, a=-20.0, b=20.0, tol=1e-14):
    """
    Minimise  g(lam) = func(X + lam*S)  over [a, b].
    Returns optimal step length lam*.
    """
    gr = (np.sqrt(5) + 1) / 2

    def g(lam):
        return func(X + lam * S)

    c = b - (b - a) / gr
    d = a + (b - a) / gr

    for _ in range(300):
        if abs(b - a) < tol:
            break
        if g(c) < g(d):
            b = d
        else:
            a = c
        c = b - (b - a) / gr
        d = a + (b - a) / gr

    return (a + b) / 2.0

def fmt(X):
    return "[" + ", ".join(f"{xi:11.7f}" for xi in X) + "]"

# powell's method 
def powell_method(func, X0, tol=1e-10, max_cycles=200, verbose=True):
    n = len(X0)
    X = np.array(X0, dtype=float)
    # initial directions = coordinate unit vectors
    S = np.eye(n)

    if verbose:
        print("=" * 78)
        print("Powell's Conjugate Direction Method")
        print("=" * 78)
        print(f"  f(x1, x2) = x1 - x2 + 2*x1^2 + x2^2 + 2*x1*x2")
        print(f"  Starting point : X0 = {fmt(X0)}")
        print("=" * 78)
        hdr = (f"{'Cycle':>5}  {'Direction':>16}  "
               f"{'lam*':>12}  {'X':>28}  {'f(X)':>14}")
        print(hdr)
        print("-" * 78)
        print(f"{'0':>5}  {'initial':>16}  {'---':>12}  "
              f"{fmt(X):>28}  {func(X):>14.8f}")

    for cycle in range(1, max_cycles + 1):
        X_cycle_start = X.copy()
        f_cycle_start = func(X)

        # minimise sequentially along all n directions 
        for i in range(n):
            lam = golden_section_search(func, X, S[i])
            X = X + lam * S[i]
            if verbose:
                print(f"{cycle:>5}  {'S_' + str(i+1):>16}  {lam:>12.8f}  "
                      f"{fmt(X):>28}  {func(X):>14.8f}")

        # build pattern direction
        S_pattern = X - X_cycle_start
        norm_p = np.linalg.norm(S_pattern)

        if norm_p < tol:
            if verbose:
                print(f"\n  >> Converged (|pattern dir| ~ 0) after "
                      f"{cycle} cycle(s).\n")
            break

        # minimise along pattern direction
        lam_p = golden_section_search(func, X, S_pattern)
        X = X + lam_p * S_pattern
        if verbose:
            print(f"{'':>5}  {'pattern':>16}  {lam_p:>12.8f}  "
                  f"{fmt(X):>28}  {func(X):>14.8f}")

        # update directions: discard S_1, shift left, insert pattern
        for i in range(n - 1):
            S[i] = S[i + 1]
        S[n - 1] = S_pattern / norm_p

        # convergence check
        f_new = func(X)
        delta_X = np.linalg.norm(X - X_cycle_start)
        delta_f = abs(f_new - f_cycle_start)

        if delta_X < tol and delta_f < tol:
            if verbose:
                print(f"\n  >> Converged after {cycle} cycle(s).")
                print(f"     |dX| = {delta_X:.2e},  |df| = {delta_f:.2e}\n")
            break

        # reset directions every n+1 cycles to avoid degeneracy
        if cycle % (n + 1) == 0:
            S = np.eye(n)
            if verbose:
                print(f"{'':>5}  {'** reset dirs **':>16}")

    return X, func(X)

# main
if __name__ == "__main__":
    X0 = np.array([0.0, 0.0])
    X_opt, f_opt = powell_method(f, X0, tol=1e-10, verbose=True)

    print("=" * 78)
    print("RESULTS")
    print("=" * 78)
    print(f"  Optimal point :  x1* = {X_opt[0]:.10f}")
    print(f"                   x2* = {X_opt[1]:.10f}")
    print(f"  Minimum value :  f*  = {f_opt:.10f}")
    print("=" * 78)

    # analytical verification
    X_exact = np.array([-1.0, 1.5])
    f_exact = -1.25

    print(f"\n  Analytical solution:")
    print(f"    X*  = [-1.0, 1.5]")
    print(f"    f*  = -1.25")
    print(f"\n  Errors:")
    print(f"    ||X_num - X_exact|| = {np.linalg.norm(X_opt - X_exact):.2e}")
    print(f"    |f_num  - f_exact|  = {abs(f_opt - f_exact):.2e}")
