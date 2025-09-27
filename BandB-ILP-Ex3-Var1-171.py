
"""

Section 5.2  Example 3 Variation 1 : 
             BandB-ILP-Ex3-Var1-171.py


                                 Disclaimer
                                 ==========

The code in this repository accompanies the article:

Title: Large-number optimization: Exact-Arithmetic Mathematical Programming 
       with integers and fractions beyond any bit limits

Author: Josef Kallrath
Section: D: Statistics and Operational Research

Journal: Mathematics (MDPI)
Special Issue: Innovations in Optimization and Operations ResearchYear: 2025

This code is provided as a supplement to the article for educational and research purposes. 
The author and publisher are not responsible for any errors or damages resulting from its use.

We encourage users to provide feedback on ideas, report any errors, or share
improvements to the code. Contributions and suggestions are welcome to enhance
the functionality and accuracy of this repository. Please submit your feedback
or proposed changes via GitHub issues or pull requests, or send an email to
the author at josef.kallrath@web.de


This implementation handles the bilinear optimization problem in Section 5.3

"""
	
  

from fractions import Fraction
import math
from typing import List, Tuple, Optional, Dict
from copy import deepcopy
import heapq
import re
from sympy import sympify
from Simplex_class import SimplexProblem

# Custom infinity handling for Fractions
POS_INF = Fraction(10**50, 1)  # Large number to represent infinity
NEG_INF = Fraction(-10**50, 1)

class Node:
    def __init__(self, lp_solution: Dict, lp_value: Fraction, bounds: List[Tuple[Optional[Fraction], Optional[Fraction]]]):
        self.lp_solution = lp_solution  # Solution to the LP relaxation
        self.lp_value = lp_value        # Objective value of the LP relaxation
        self.bounds = bounds            # Variable bounds for this node

def parse_expression(expr):
    """Parse a string expression into a Fraction.
    Handles arithmetic operations, exponents, and parentheses.
    """
    if expr is None:
        return None
    
    expr_str = str(expr).replace(' ', '')
    
    # Handle parentheses recursively
    while '(' in expr_str and ')' in expr_str:
        start = expr_str.rfind('(')
        end = expr_str.find(')', start)
        if start == -1 or end == -1:
            break
        sub_expr = expr_str[start+1:end]
        sub_value = parse_expression(sub_expr)
        expr_str = expr_str[:start] + str(sub_value) + expr_str[end+1:]
    
    try:
        # Handle exponents first (convert ^ to **)
        if '^' in expr_str:
            expr_str = expr_str.replace('^', '**')
        if '**' in expr_str:
            base, exponent = expr_str.split('**', 1)
            base_val = parse_expression(base)
            exponent_val = parse_expression(exponent)
            return Fraction(int(float(base_val) ** float(exponent_val)))
            
        # Handle multiplication and division
        if '*' in expr_str or '/' in expr_str:
            parts = re.split('([*/])', expr_str)
            result = parse_expression(parts[0])
            for i in range(1, len(parts), 2):
                op = parts[i]
                num = parse_expression(parts[i+1])
                if op == '*':
                    result *= num
                else:
                    result /= num
            return result
            
        # Handle addition and subtraction
        if '+' in expr_str or ('-' in expr_str and len(expr_str) > 1 and not expr_str.startswith('-')):
            parts = re.split('([+-])', expr_str)
            if parts[0] == '':
                parts = parts[1:]
                parts[0] = '-' + parts[0]
            result = parse_expression(parts[0])
            for i in range(1, len(parts), 2):
                op = parts[i]
                num = parse_expression(parts[i+1])
                if op == '+':
                    result += num
                else:
                    result -= num
            return result
            
        # Handle simple fractions
        if '/' in expr_str:
            num, denom = expr_str.split('/')
            return Fraction(int(num), int(denom))
        return Fraction(int(expr_str))
        
    except Exception as e:
        raise ValueError(f"Could not parse expression: {expr_str}. Error: {str(e)}")

class MILPSolver:
    def __init__(self, c: List[Fraction], A: List[List[Fraction]], b: List[Fraction], 
                 integer_vars: List[int], bounds: List[Tuple[Optional[Fraction], Optional[Fraction]]],
                 sense: str = 'min', Print_Debug: int = 0):
        """
        Initialize the MILP solver with exact fractional arithmetic.
        
        Args:
            c: Objective coefficients (n elements)
            A: Constraint matrix (m x n)
            b: Right-hand side (m elements)
            integer_vars: Indices of variables that must be integer
            bounds: List of (lower_bound, upper_bound) for each variable
            sense: 'min' or 'max'
            Print_Debug: Debug level (0=none, 1=basic, 2=detailed, 3=verbose)
        """
        self.c = c
        self.A = A
        self.b = b
        self.integer_vars = integer_vars
        self.initial_bounds = bounds
        self.sense = sense
        self.Print_Debug = Print_Debug
        self.n = len(c)
        self.m = len(b)
        
        # Best solution found so far
        self.best_solution = None
        self.best_value = POS_INF if sense == 'min' else NEG_INF
        
        # For tracking progress
        self.nodes_explored = 0
    
    def solve_lp_relaxation(self, bounds: List[Tuple[Optional[Fraction], Optional[Fraction]]]) -> Tuple[Optional[Dict], Fraction, str]:
        """
        Solve the LP relaxation using the exact simplex method.
        Returns:
            Tuple containing:
            - Solution dictionary (variable index to value)
            - Objective value
            - Problem status ('optimal', 'infeasible', 'unbounded')
        """
        # Convert bounds to the format expected by SimplexProblem
        simplex_bounds = [(str(lo) if lo is not None else None, 
                          str(hi) if hi is not None else None) 
                         for lo, hi in bounds]
        
        # Create and solve the LP problem
        problem = SimplexProblem(
            c=[str(x) for x in self.c],
            A=[[str(x) for x in row] for row in self.A],
            b=[str(x) for x in self.b],
            rel=['<='] * self.m,  # Assuming all constraints are <= for simplicity
            bounds=simplex_bounds,
            sense=self.sense,
            Print_Debug=max(0, self.Print_Debug - 1)  # Reduce debug level by 1
        )
        
        solution, obj_value, status = problem.solve()
        
        if status != 'optimal':
            return None, POS_INF if self.sense == 'min' else NEG_INF, status
        
        # Convert solution back to Fraction format
        lp_solution = {i: Fraction(str(solution[i])) for i in range(self.n)}
        lp_value = Fraction(str(obj_value))
        
        if self.Print_Debug >= 3:
            print("\nLP Solution:")
            for var in range(self.n):
                print(f"  x{var} = {lp_solution.get(var, 0)} (bounds: {bounds[var]})")
            print(f"  Objective = {lp_value}")
            print(f"  Status = {status}")
        
        return lp_solution, lp_value, status
    
    def is_integer_feasible(self, solution: Dict) -> bool:
        """Check if the solution satisfies all integer constraints."""
        for var in self.integer_vars:
            if var in solution and not solution[var].denominator == 1:
                return False
        return True
    
    def branch(self, node: Node) -> List[Node]:
        """Branch on a fractional variable by adjusting bounds."""
        children = []
        
        # Find the most fractional variable among integer variables
        max_frac = 0
        branching_var = None
        
        for var in self.integer_vars:
            val = node.lp_solution[var]
            if val.denominator != 1:
                frac = abs(val - round(float(val)))
                if frac > max_frac:
                    max_frac = frac
                    branching_var = var
        
        if branching_var is None:
            return children
        
        val = node.lp_solution[branching_var]
        floor_val = Fraction(int(val))  # Floor
        ceil_val = floor_val + 1       # Ceiling
        
        if self.Print_Debug >= 2:
            print(f"\nBranching on x{branching_var} = {val} (between {floor_val} and {ceil_val})")
        
        # Create two child nodes by adjusting bounds
        # Left child: x <= floor_val
        left_bounds = deepcopy(node.bounds)
        old_lo, old_hi = left_bounds[branching_var]
        left_bounds[branching_var] = (old_lo, min(old_hi, floor_val) if old_hi is not None else floor_val)
        
        left_sol, left_val, left_status = self.solve_lp_relaxation(left_bounds)
        if left_status == 'optimal':
            children.append(Node(left_sol, left_val, left_bounds))
            if self.Print_Debug >= 2:
                print(f"Created left child with x{branching_var} <= {floor_val}: obj = {left_val}")
        
        # Right child: x >= ceil_val
        right_bounds = deepcopy(node.bounds)
        old_lo, old_hi = right_bounds[branching_var]
        right_bounds[branching_var] = (max(old_lo, ceil_val) if old_lo is not None else ceil_val, old_hi)
        
        right_sol, right_val, right_status = self.solve_lp_relaxation(right_bounds)
        if right_status == 'optimal':
            children.append(Node(right_sol, right_val, right_bounds))
            if self.Print_Debug >= 2:
                print(f"Created right child with x{branching_var} >= {ceil_val}: obj = {right_val}")
        
        return children
    
    def should_prune(self, node: Node) -> bool:
        """Determine if a node should be pruned."""
        if self.sense == 'min':
            return node.lp_value >= self.best_value
        else:
            return node.lp_value <= self.best_value
    
    def solve(self) -> Tuple[Optional[Dict], Fraction]:
        """Main B&B algorithm using exact simplex for LP relaxations."""
        # Priority queue for nodes (using LP value)
        heap = []
        
        # Initial node with original bounds
        initial_sol, initial_val, initial_status = self.solve_lp_relaxation(self.initial_bounds)
        if initial_status != 'optimal':
            if self.Print_Debug >= 1:
                print("Initial LP relaxation is infeasible or unbounded")
            return None, POS_INF if self.sense == 'min' else NEG_INF
        
        initial_node = Node(initial_sol, initial_val, deepcopy(self.initial_bounds))
        
        if self.Print_Debug >= 1:
            print("\nInitial LP relaxation:")
            for var in range(self.n):
                print(f"  x{var} = {initial_sol.get(var, 0)} (bounds: {self.initial_bounds[var]})")
            print(f"  Objective = {initial_val}")
        
        # Use max heap for maximization, min heap for minimization
        if self.sense == 'min':
            heapq.heappush(heap, (initial_val, self.nodes_explored, initial_node))
        else:
            heapq.heappush(heap, (-initial_val, self.nodes_explored, initial_node))
        self.nodes_explored += 1
        
        while heap:
            _, _, node = heapq.heappop(heap)
            
            if self.Print_Debug >= 2:
                print(f"\nProcessing node {node.bounds}:")
                print(f"  LP value: {node.lp_value}")
                for var in range(self.n):
                    print(f"  x{var} = {node.lp_solution.get(var, 0)} (bounds: {node.bounds[var]})")
            
            # Prune if worse than current best
            if self.should_prune(node):
                if self.Print_Debug >= 2:
                    print("Pruning this node (bound worse than current best)")
                continue
            
            # Check for integer feasibility
            if self.is_integer_feasible(node.lp_solution):
                if (self.sense == 'min' and node.lp_value < self.best_value) or \
                   (self.sense == 'max' and node.lp_value > self.best_value):
                    self.best_value = node.lp_value
                    self.best_solution = node.lp_solution
                    if self.Print_Debug >= 1:
                        print(f"\nNew incumbent solution found:")
                        for var in range(self.n):
                            print(f"  x{var} = {node.lp_solution.get(var, 0)} (bounds: {node.bounds[var]})")
                        print(f"  Objective = {node.lp_value}")
                continue
            
            # Branch
            children = self.branch(node)
            for child in children:
                self.nodes_explored += 1
                if self.sense == 'min':
                    heapq.heappush(heap, (child.lp_value, self.nodes_explored, child))
                else:
                    heapq.heappush(heap, (-child.lp_value, self.nodes_explored, child))
        
        return self.best_solution, self.best_value

if __name__ == "__main__":
    # Example problem
    N = 10**100
    c = [1+N, 1+N, 2+N]
    A = [
        [7, 2, 3],
        [5, 4, 7],
        [2, 3, 5]
    ]
    b = [26, 42, 28]
    integer_vars = [0, 1, 2]  # all three variables are integer
    bounds = [(0, 4), (0, 10), (0, 6)]  # Bounds for x1, x2, x3
    
    solver = MILPSolver(
        c=[Fraction(x) for x in c],
        A=[[Fraction(x) for x in row] for row in A],
        b=[Fraction(x) for x in b],
        integer_vars=integer_vars,
        bounds=bounds,
        sense='max',
        Print_Debug=3
    )
    solution, value = solver.solve()
    
    print("\nOptimal Solution:")
    if solution is None:
        print("No feasible solution found")
    else:
        for var in range(len(solution)):
            print(f"x{var} = {solution.get(var, 0)}")
        print(f"Objective value: {value}")
    print(f"Nodes explored: {solver.nodes_explored}")
