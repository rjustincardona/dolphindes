"""
Dual Problem Interface

"""

__all__ = ['SharedProjQCQP'] 

import numpy as np 

class SharedProjQCQP():
    def __init__(self):
        const_objective = None 
        linear_objective = None
        quadratic_objective = None

        projections = [] 

    def get_dual(self):
        pass 

    def solve_current_problem(self):
        pass 

    def solve_gcd(self):
        pass 

