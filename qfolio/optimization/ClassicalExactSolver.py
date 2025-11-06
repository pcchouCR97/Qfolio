"""
This module provides a classical, exact QUBO solver that is compatible with
the Qiskit optimization framework.

It uses Gurobi for solving quadratic binary programs.
This provides a deterministic and exact solution, guaranteeing optimality.

To use this solver, you must have the required packages installed:
pip install gurobipy

Note: Gurobi is free for academic use. Get a license at: https://www.gurobi.com/academia/
For non-academic use, this solver will work in demo mode (limited problem sizes).
"""

import numpy as np
from qiskit_optimization.algorithms import OptimizationResult, SolutionSample, OptimizationResultStatus
from qiskit_optimization import QuadraticProgram
import gurobipy as gp
from gurobipy import GRB

class ExactMIQPSolver:
    """
    A wrapper for Gurobi that is compatible with Qiskit Optimization framework.
    """
    def solve(self, qp: QuadraticProgram) -> OptimizationResult:
        """
        Solves a Qiskit QuadraticProgram using Gurobi.

        Args:
            qp: The QuadraticProgram to be solved.

        Returns:
            An OptimizationResult object containing the solution.
        """
        num_vars = qp.get_num_vars()

        # Create Gurobi model
        model = gp.Model("portfolio_optimization")
        model.setParam('OutputFlag', 0)  # Suppress output

        # Create binary variables
        vars_list = model.addVars(num_vars, vtype=GRB.BINARY, name="x")

        # Build quadratic objective
        obj_expr = gp.QuadExpr()

        # Add quadratic terms
        if qp.objective.quadratic is not None:
            quadratic_obj = qp.objective.quadratic.to_dict(use_name=False)
            for (i, j), coeff in quadratic_obj.items():
                obj_expr += coeff * vars_list[i] * vars_list[j]

        # Add linear terms
        if qp.objective.linear is not None:
            linear_obj = qp.objective.linear.to_dict(use_name=False)
            for i, coeff in linear_obj.items():
                obj_expr += coeff * vars_list[i]

        # Add constant offset
        if qp.objective.constant is not None:
            obj_expr += qp.objective.constant

        # Set objective (minimize)
        model.setObjective(obj_expr, GRB.MINIMIZE)

        # Add constraints
        for constraint in qp.linear_constraints:
            lhs = constraint.linear.to_dict(use_name=False)
            lhs_expr = gp.LinExpr()
            for i, coeff in lhs.items():
                lhs_expr += coeff * vars_list[i]

            if constraint.sense.name == 'LE':  # <=
                model.addConstr(lhs_expr <= constraint.rhs)
            elif constraint.sense.name == 'GE':  # >=
                model.addConstr(lhs_expr >= constraint.rhs)
            elif constraint.sense.name == 'EQ':  # ==
                model.addConstr(lhs_expr == constraint.rhs)

        # Solve
        try:
            model.optimize()

            # Check status
            if model.status == GRB.OPTIMAL:
                solution_status = OptimizationResultStatus.SUCCESS
                # Extract solution
                x_solution = [round(vars_list[i].X) for i in range(num_vars)]
                fval = model.ObjVal
            else:
                print(f"Warning: Gurobi did not find optimal solution. Status: {model.status}")
                solution_status = OptimizationResultStatus.FAILURE
                x_solution = [0] * num_vars
                fval = 0.0

        except Exception as e:
            print(f"Gurobi SolverError: {e}")
            print(f"Please ensure Gurobi is installed: pip install gurobipy")
            print(f"For academic use, get a free license at: https://www.gurobi.com/academia/")
            solution_status = OptimizationResultStatus.FAILURE
            x_solution = [0] * num_vars
            fval = 0.0

        qiskit_solution = SolutionSample(
            x=np.array(x_solution),
            fval=fval,
            probability=1.0,
            status=solution_status
        )

        return OptimizationResult(
            x=qiskit_solution.x,
            fval=qiskit_solution.fval,
            variables=qp.variables,
            status=qiskit_solution.status,
            raw_results=model,
            samples=[qiskit_solution]
        )
