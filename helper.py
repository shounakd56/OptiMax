import numpy as np
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
import cvxopt as opt
from cvxopt import solvers
from scipy.optimize import NonlinearConstraint

def portfolio_return(weights, mean_returns):
    return -np.dot(weights, mean_returns)  # Negative for maximization

def portfolio_risk(weights, cov_matrix):
    return np.sqrt(weights.T @ cov_matrix @ weights)

def diversification_constraint(weights, max_weight):
    return max_weight - np.max(weights)

# Optimization using SLSQP
def optimize_slsqp(mean_returns, cov_matrix, max_risk, max_weight):
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Sum of weights = 1
        {'type': 'ineq', 'fun': lambda x: max_risk - portfolio_risk(x, cov_matrix)},
        {'type': 'ineq', 'fun': lambda x: diversification_constraint(x, max_weight)}
    ]
    bounds = [(0, max_weight) for _ in range(len(mean_returns))]
    initial_weights = np.ones(len(mean_returns)) / len(mean_returns)

    result = minimize(
        portfolio_return,
        x0=initial_weights,
        args=(mean_returns,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    return result.x, -result.fun, portfolio_risk(result.x, cov_matrix)

# Optimization using Quadratic Programming (QP)
def optimize_qp(mean_returns, cov_matrix,max_weight):
    n = len(mean_returns)
    P = opt.matrix(cov_matrix.values)
    q = opt.matrix(-mean_returns)
    G = opt.matrix(np.vstack([np.eye(n), -np.eye(n)]))
    h = opt.matrix(np.hstack([np.ones(n) * max_weight, np.zeros(n)]))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)

    sol = solvers.qp(P, q, G, h, A, b)
    weights = np.array(sol['x']).flatten()
    return weights, np.dot(weights, mean_returns), portfolio_risk(weights, cov_matrix)

# Optimization using Differential Evolution (DE)
def optimize_de(mean_returns, cov_matrix, max_risk, max_weight):
    bounds = [(0, max_weight) for _ in range(len(mean_returns))]
    

    def objective(weights):
        return -np.dot(weights, mean_returns)


    def risk_constraint(weights):
        return max_risk - portfolio_risk(weights, cov_matrix)


    def weight_sum_constraint(weights):
        return np.sum(weights) - 1

    risk_nonlinear_constraint = NonlinearConstraint(
        fun=risk_constraint,
        lb=0,  
        ub=np.inf
    )


    weight_sum_nonlinear_constraint = NonlinearConstraint(
        fun=weight_sum_constraint,
        lb=0,
        ub=0
    )

    result = differential_evolution(
        objective,
        bounds=bounds,
        constraints=(risk_nonlinear_constraint, weight_sum_nonlinear_constraint),
        strategy='best1bin',
        maxiter=1000
    )
    
    weights = result.x
    return weights, np.dot(weights, mean_returns), portfolio_risk(weights, cov_matrix)

# Function to calculate cumulative returns
def calculate_cumulative_returns(portfolio_returns):
    return (1 + portfolio_returns).cumprod()
