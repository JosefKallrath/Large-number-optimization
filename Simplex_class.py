"""
Exact Fraction Linear Programming Solver using Two-Phase Simplex Method

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


This implementation handles:
- Exact arithmetic using fractions (no floating-point rounding errors)
- Both minimization and maximization problems
- Lower and upper bounds on variables
- Equality and inequality constraints
- Two-phase simplex method with artificial variables
- Proper transformation of bounded variables
- Option to convert upper bounds to <= inequality constraints (CP_UB_as_LE)
- Option to convert lower bounds to >= inequality constraints (CP_LB_treatment)

Key Components:
1. parse_expression: Converts string expressions to exact Fractions
2. SimplexProblem class: Main solver class with these methods:
   - __init__: Initializes and transforms the problem
   - simplex_kernel: Core simplex algorithm implementation
   - create_phase1_problem: Sets up Phase 1 problem with artificial variables
   - create_phase2_problem: Sets up Phase 2 problem after finding feasible solution
   - solve: Manages the two-phase solving process with improved basis cleaning
   - print_problem: Displays problem formulation
   - _ensure_fraction: Auxiliary code for exact arithmetic
The solver transforms variables with lower bounds L using x = x' + L when CP_LB_treatment=2,
otherwise treats them as constraints when CP_LB_treatment=1. This transformation is handled
in the Presolve method, which now displays transformed constraints. Enhanced basis cleaning
ensures robust handling of degenerate cases for feasibility.
"""

from fractions import Fraction
import re
from sympy import sympify

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

class SimplexProblem:
    """Main class for solving LP problems using exact fraction arithmetic."""
    
    def __init__(self, c, A, b, rel, bounds, sense, Print_Debug=0, CP_UB_as_LE=1, CP_LB_treatment=2):
        # Save original problem data immediately
        self.original_data = {
            'c': c.copy(),
            'A': [row.copy() for row in A],
            'b': b.copy(),
            'rel': rel.copy(),
            'bounds': bounds.copy(),
            'sense': sense
        }

        # Create P-data for presolve and later transformations
        self.P_data = {
            'c': None,
            'A': None,
            'b': None,
            'rel': None,
            'bounds': None,
            'num_vars': None,
            'num_constraints': None
        }
    
        """Initialize and transform the LP problem.
        
        Args:
            c: List of objective coefficients
            A: Constraint matrix (list of lists)
            b: Right-hand side values
            rel: List of constraint relations ('<=', '>=', '=')
            bounds: List of (lower_bound, upper_bound) tuples
            sense: 'max' or 'min'
            Print_Debug: Debug level (0-3)
            CP_UB_as_LE: Control parameter (0/1/2) 
                        0: keep bounds as bounds
                        1: convert upper bounds to <= constraints
                        2: convert all bounds to constraints
            CP_LB_treatment: Control parameter (0/1/2)
                        0: treat lower bounds as bounds
                        1: convert lower bounds to >= constraints
                        2: transform x = L + x' (default)
        """
        self.sense = sense.lower()
        if self.sense not in ('max', 'min'):
            raise ValueError("Sense must be either 'max' or 'min'")
            
        self.Print_Debug = Print_Debug
        self.CP_UB_as_LE = CP_UB_as_LE
        self.CP_LB_treatment = CP_LB_treatment
        self.original_num_vars = len(c)
        self.num_constraints = len(A)
       
        def to_expr(x):
            """Convert input to sympy expression for exact arithmetic."""
            if isinstance(x, str):
                return sympify(x.replace('^', '**'))
            return sympify(x)

        # Store original problem data before transformation
        self.original_c = [to_expr(x) for x in c]
        self.original_A = [[to_expr(x) for x in row] for row in A]
        self.original_b = [to_expr(x) for x in b]
        self.original_rel = rel.copy()
        self.original_bounds = bounds.copy()
        
        # Track flipped equality constraints
        self.eq_flip_flags = [False] * self.num_constraints
        
        # Parse bounds
        self.lower_bounds = [parse_expression(lo) if lo is not None else Fraction(0, 1) for lo, _ in bounds]
        self.upper_bounds = [parse_expression(hi) if hi is not None else None for _, hi in bounds]
        
        # Initialize problem data (no lower bound transformation here)
        self.c = [self._ensure_fraction(x) for x in self.original_c]
        self.A = [[self._ensure_fraction(x) for x in row] for row in self.original_A]
        self.b = [self._ensure_fraction(x) for x in self.original_b]
        self.rel = self.original_rel.copy()
        self.bounds = [(Fraction(0, 1), hi) for hi in self.upper_bounds]  # Lower bounds set to 0 initially
        
        self.num_vars = len(self.c)
        self.var_names = [f'x{i+1}' for i in range(self.num_vars)]
        self.var_types = ['decision'] * self.num_vars

        if self.Print_Debug >= 0:
            self.print_problem()

    def simplex_kernel(self, c, A, b, basis, sense, phase=1):
        """Core simplex algorithm implementation with exact Fractions.
        
        Args:
            c: Objective coefficients
            A: Constraint matrix
            b: Right-hand side values
            basis: Initial basis indices
            sense: 'min' or 'max'
            phase: 1 or 2 (for debugging)
        
        Returns:
            Final tableau and basis
        """
        # Convert all inputs to exact Fractions
        tableau = [
            [self._ensure_fraction(x) for x in row] + [self._ensure_fraction(b[i])]
            for i, row in enumerate(A)
        ]
        tableau.append([self._ensure_fraction(x) for x in c] + [Fraction(0, 1)])

        iteration = 0  # Initialize iteration counter
        max_iterations = 1000
        
        while True:
            iteration += 1
            
            if self.Print_Debug >= 3:
                print(f"\n--- Iteration {iteration} (Phase {phase}) ---")
                print("Basis:", [self.var_names[i] for i in basis])
                print("Solution:", [tableau[i][-1] for i in range(len(basis))])
                print("Reduced Costs:")
                for j in range(len(tableau[0])-1):
                    if j not in basis:
                        print(f"{self.var_names[j]}: {tableau[-1][j]}")

            # Optimality check with exact Fraction comparison
            is_optimal = True
            for j in range(len(tableau[0])-1):
                if j not in basis:
                    rc = tableau[-1][j]
                    if (sense == 'min' and rc < 0) or (sense == 'max' and rc > 0):
                        is_optimal = False
                        break
            
            if is_optimal:
                # Enhanced basis cleaning for Phase 1 when all reduced costs are zero
                if phase == 1:
                    artificial_in_basis = [i for i, var_idx in enumerate(basis) 
                                        if var_idx < len(self.var_types) and 
                                        self.var_types[var_idx] == 'artificial']
                    
                    if artificial_in_basis:
                        if self.Print_Debug >= 2:
                            print("\nPhase 1: All reduced costs zero but artificials in basis")
                            print("Attempting multi-stage pivot out of artificial variables...")
                        
                        # Stage 1: Try all possible pivots for each artificial variable
                        made_progress = True
                        while made_progress and artificial_in_basis:
                            made_progress = False
                            new_artificial_in_basis = []
                            
                            for row_idx in artificial_in_basis:
                                var_idx = basis[row_idx]
                                if self.Print_Debug >= 3:
                                    print(f"  Processing {self.var_names[var_idx]} in row {row_idx}")
                                
                                # Find all possible entering variables
                                candidates = []
                                for j in range(len(tableau[0])-1):
                                    if (j not in basis and 
                                        tableau[row_idx][j] != 0 and 
                                        self.var_types[j] != 'artificial'):
                                        candidates.append(j)
                                
                                # Sort by preference: decision variables first, then slack/surplus
                                candidates.sort(key=lambda x: (
                                    0 if self.var_types[x] == 'decision' else 1, x))
                                
                                # Try each candidate exhaustively
                                pivot_success = False
                                for entering_col in candidates:
                                    if tableau[row_idx][entering_col] == 0:
                                        continue
                                    
                                    if self.Print_Debug >= 3:
                                        print(f"    Attempting pivot with {self.var_names[entering_col]}")
                                    
                                    # Test the pivot
                                    test_tableau = [row.copy() for row in tableau]
                                    test_basis = basis.copy()
                                    
                                    pivot_val = test_tableau[row_idx][entering_col]
                                    for k in range(len(test_tableau[row_idx])):
                                        test_tableau[row_idx][k] /= pivot_val
                                    
                                    for k in range(len(test_tableau)):
                                        if k != row_idx:
                                            factor = test_tableau[k][entering_col]
                                            for l in range(len(test_tableau[k])):
                                                test_tableau[k][l] -= factor * test_tableau[row_idx][l]
                                    
                                    test_basis[row_idx] = entering_col
                                    
                                    # Check validity (non-negative RHS)
                                    valid = True
                                    for i in range(len(test_basis)):
                                        if test_tableau[i][-1] < -Fraction(1, 1000000):
                                            valid = False
                                            break
                                    
                                    if valid:
                                        if self.Print_Debug >= 2:
                                            print(f"    Successful pivot: {self.var_names[var_idx]} -> {self.var_names[entering_col]}")
                                        tableau = test_tableau
                                        basis = test_basis
                                        pivot_success = True
                                        made_progress = True
                                        break  # Move to next artificial variable
                                    elif self.Print_Debug >= 3:
                                        print(f"    Pivot failed (infeasible RHS)")
                                
                                if not pivot_success:
                                    new_artificial_in_basis.append(row_idx)
                            
                            artificial_in_basis = new_artificial_in_basis
                        
                        # Final check for remaining artificial variables
                        if artificial_in_basis:
                            all_zero = True
                            for row_idx in artificial_in_basis:
                                if abs(tableau[row_idx][-1]) > Fraction(1, 1000000):
                                    all_zero = False
                                    break
                            
                            if all_zero:
                                if self.Print_Debug >= 1:
                                    print("Note: Some artificial variables remain with zero values")
                            else:
                                if self.Print_Debug >= 1:
                                    print("Could not eliminate all artificial variables")
                                raise ValueError("Problem is infeasible")
                
                if is_optimal:
                    if self.Print_Debug >= 1:
                        print(f"Optimal after {iteration} iterations (Phase {phase})")
                    break

            if iteration >= max_iterations:
                raise ValueError("Max iterations exceeded")

            # Pivot selection: find entering variable
            entering_col = -1
            best_rc = Fraction(0, 1)
            for j in range(len(tableau[0])-1):
                if j not in basis:
                    rc = tableau[-1][j]
                    if (sense == 'min' and rc < best_rc) or (sense == 'max' and rc > best_rc):
                        best_rc = rc
                        entering_col = j

            if entering_col == -1:
                break

            # Ratio test: find leaving variable
            leaving_row = -1
            min_ratio = None
            for i in range(len(tableau)-1):
                a_ij = tableau[i][entering_col]
                if a_ij <= 0:
                    continue
                ratio = tableau[i][-1] / a_ij
                if min_ratio is None or ratio < min_ratio:
                    min_ratio = ratio
                    leaving_row = i

            if leaving_row == -1:
                raise ValueError("Problem is unbounded")

            # Pivot operation
            pivot_val = tableau[leaving_row][entering_col]
            for j in range(len(tableau[leaving_row])):
                tableau[leaving_row][j] /= pivot_val
            
            for i in range(len(tableau)):
                if i != leaving_row:
                    factor = tableau[i][entering_col]
                    for j in range(len(tableau[i])):
                        tableau[i][j] -= factor * tableau[leaving_row][j]

            basis[leaving_row] = entering_col

        return tableau, basis


    def create_phase1_problem(self):
        """Create Phase 1 problem with artificial variables.
        
        Returns:
            phase1_c: Phase 1 objective coefficients
            phase1_A: Phase 1 constraint matrix
            phase1_b: Phase 1 right-hand side
            phase1_basis: Initial basis indices
        """
        # First check if any variable's lower bound > upper bound (infeasible)
        for j, (lo, hi) in enumerate(self.bounds):
            if lo is not None and hi is not None:
                lo_frac = self._ensure_fraction(lo)
                hi_frac = self._ensure_fraction(hi)
                if lo_frac > hi_frac:
                    raise ValueError(f"Problem is infeasible - variable x{j+1} has lower bound {lo_frac} > upper bound {hi_frac}")
        
        # Count needed slack/surplus/artificial variables
        num_slack = sum(1 for r in self.rel if r == '<=')
        num_surplus = sum(1 for r in self.rel if r == '>=')
        num_artificial = sum(1 for r in self.rel if r in ('>=', '='))
        
        # Total variables: original + slack/surplus + artificial
        total_vars = self.num_vars + num_slack + num_surplus + num_artificial
        
        # Create Phase 1 objective (minimize sum of artificial vars)
        phase1_c = [Fraction(0, 1) for _ in range(total_vars)]
        
        # Artificial variables start after original + slack/surplus
        artificial_start = self.num_vars + num_slack + num_surplus
        for j in range(artificial_start, total_vars):
            phase1_c[j] = Fraction(1, 1)
        
        # Create constraint matrix
        phase1_A = []
        slack_idx = self.num_vars
        surplus_idx = self.num_vars + num_slack
        artificial_idx = surplus_idx + num_surplus
        
        slack_count = 0
        surplus_count = 0
        artificial_count = 0
        
        # Update variable names and types
        self.var_names = [f'x{i+1}' for i in range(self.num_vars)]
        self.var_types = ['decision'] * self.num_vars
        
        # Add slack/surplus/artificial variables to names
        for i in range(num_slack):
            self.var_names.append(f's_{i+1}')
            self.var_types.append('slack')
        for i in range(num_surplus):
            self.var_names.append(f'e_{i+1}')
            self.var_types.append('surplus')
        for i in range(num_artificial):
            self.var_names.append(f'a_{i+1}')
            self.var_types.append('artificial')
        
        # Process each constraint
        phase1_b = [self._ensure_fraction(x) for x in self.b]
        for i in range(self.num_constraints):
            row = [Fraction(0, 1) for _ in range(total_vars)]
            
            # Original variables
            for j in range(self.num_vars):
                row[j] = self._ensure_fraction(self.A[i][j])
            
            # Add slack/surplus/artificial variables
            if self.rel[i] == '<=':
                row[slack_idx + slack_count] = Fraction(1, 1)
                slack_count += 1
            elif self.rel[i] == '>=':
                row[surplus_idx + surplus_count] = Fraction(-1, 1)
                row[artificial_idx + artificial_count] = Fraction(1, 1)
                surplus_count += 1
                artificial_count += 1
            elif self.rel[i] == '=':
                row[artificial_idx + artificial_count] = Fraction(1, 1)
                artificial_count += 1
            
            phase1_A.append(row)
        
        # Initial basis consists of slack and artificial variables
        basis = []
        slack_count = 0
        artificial_count = 0
        
        for i in range(self.num_constraints):
            if self.rel[i] == '<=':
                basis.append(slack_idx + slack_count)
                slack_count += 1
            else:
                basis.append(artificial_idx + artificial_count)
                artificial_count += 1
        
        return phase1_c, phase1_A, phase1_b, basis


    def create_phase2_problem(self, phase1_tableau, phase1_basis):
        """Create Phase 2 problem with robust Fraction handling and correct dual variables."""
        # 1. Clean the basis and tableau - ensure all values are Fractions
        clean_basis = []
        clean_tableau = []
        
        for i, var_idx in enumerate(phase1_basis):
            if var_idx >= len(self.var_types) or self.var_types[var_idx] == 'artificial':
                if self.Print_Debug >= 2:
                    print(f"Skipping artificial variable {var_idx} in basis cleaning")
                continue
                
            clean_row = [self._ensure_fraction(x) for x in phase1_tableau[i]]
            clean_tableau.append(clean_row)
            clean_basis.append(var_idx)
            
            if self.Print_Debug >= 3:
                print(f"Keeping basic variable {self.var_names[var_idx]} at index {var_idx}")
    
        # Ensure we have enough basic variables
        if len(clean_basis) < self.num_constraints:
            if self.Print_Debug >= 1:
                print(f"Warning: Only {len(clean_basis)} basic variables after cleaning, need {self.num_constraints}")
            
            # Add slack/surplus variables to complete the basis if needed
            for j in range(len(self.var_types)):
                if (j not in clean_basis and 
                    j < len(self.var_types) and 
                    self.var_types[j] in ('slack', 'surplus')):
                    clean_basis.append(j)
                    if self.Print_Debug >= 2:
                        print(f"Adding slack/surplus variable {self.var_names[j]} to basis")
                    if len(clean_basis) == self.num_constraints:
                        break
    
        # Identify variables to keep (decision + slack/surplus)
        keep_vars = [j for j, var_type in enumerate(self.var_types) 
                    if var_type in ('decision', 'slack', 'surplus') and j < len(self.var_types)]
        
        # Robust column mapping
        col_mapping = {}
        valid_columns = 0
        for j in keep_vars:
            if j < len(phase1_tableau[0]):
                col_mapping[j] = valid_columns
                valid_columns += 1
    
        # Calculate dual variables
        dual_vars = []
        for i, basic_var in enumerate(clean_basis):
            if basic_var < len(self.c):
                dual_val = self._ensure_fraction(self.c[basic_var])
                if self.eq_flip_flags[i] and self.sense == 'min':
                    dual_val = -dual_val
            else:
                dual_val = Fraction(0, 1)
            dual_vars.append(dual_val)
    
        # Compute reduced costs
        phase2_c = [Fraction(0, 1) for _ in keep_vars]
        
        for j in keep_vars:
            if j < len(self.c):
                phase2_c[col_mapping[j]] = self._ensure_fraction(self.c[j])
        
        for i in range(len(clean_basis)):
            basic_var = clean_basis[i]
            if basic_var in keep_vars:
                for j in range(len(phase2_c)):
                    multiplier = -1 if self.eq_flip_flags[i] else 1
                    phase2_c[j] -= dual_vars[i] * clean_tableau[i][keep_vars[j]] * multiplier
    
        # Build Phase 2 problem
        phase2_A = []
        phase2_b = []
        for i in range(len(clean_tableau)):
            row = [self._ensure_fraction(clean_tableau[i][j]) for j in keep_vars]
            if self.eq_flip_flags[i]:
                row = [-x for x in row]
            phase2_A.append(row)
            phase2_b.append(self._ensure_fraction(clean_tableau[i][-1]))
    
        if self.Print_Debug >= 3:
            print("\nPhase 2 constraint matrix:")
            for i, row in enumerate(phase2_A):
                print(f"Constraint {i+1}: {row} {self.rel[i]} {phase2_b[i]}")
    
        # Rebuild basis
        phase2_basis = [col_mapping[var] for var in clean_basis if var in col_mapping]
    
        missing_basics = self.num_constraints - len(phase2_basis)
        if missing_basics > 0:
            if self.Print_Debug >= 2:
                print(f"Need to add {missing_basics} more variables to basis")
            
            for j in range(len(keep_vars)):
                if missing_basics <= 0:
                    break
                if j not in phase2_basis:
                    phase2_basis.append(j)
                    missing_basics -= 1
                    if self.Print_Debug >= 2:
                        print(f"Added variable {self.var_names[keep_vars[j]]} to basis")
    
        if self.Print_Debug >= 1:
            print("\nFinal Phase 2 basis:")
            for basic in phase2_basis:
                print(f"  {self.var_names[keep_vars[basic]]}")
            print("Phase 2 problem dimensions:")
            print(f"  Variables: {len(phase2_c)}, Constraints: {len(phase2_A)}")
    
        return phase2_c, phase2_A, phase2_b, phase2_basis


    def _ensure_fraction(self, x):
        """Convert input to Fraction while preserving exact values.
        
        Args:
            x: Input value (Fraction, int, str, or sympy expression)
            
        Returns:
            Exact Fraction representation
        """
        if isinstance(x, Fraction):
            return x
        if isinstance(x, (int, str)):
            return Fraction(str(x))
        if hasattr(x, 'evalf'):
            return Fraction(str(x))
        raise ValueError(f"Cannot convert {type(x)} to Fraction")

    def Presolve(self, convert_upper_bounds=False, convert_all_bounds=False):
        """Preprocess the LP problem with proper bound conversion handling.
        This is the ONLY place where bounds are converted to constraints and lower bound transformation occurs.
        Displays transformed constraints after lower bound transformation.
        """
        if self.Print_Debug >= 1:
            print("\n=== Running Presolve ===")
    
        # Save original problem data
        if not hasattr(self, '_original_presolve_data'):
            self._original_presolve_data = {
                'A': [row.copy() for row in self.A],
                'b': [self._ensure_fraction(x) for x in self.b],
                'rel': self.rel.copy(),
                'bounds': [tuple(b) for b in self.bounds],
                'num_vars': self.num_vars
            }
    
        # 1. Transform lower bounds if CP_LB_treatment=2 (x = x' + L)
        transformed = False
        if self.CP_LB_treatment == 2:
            if self.Print_Debug >= 1:
                print("Transforming variables with lower bounds (x = x' + L)")
            new_A = []
            new_b = []
            new_bounds = []
            
            for i in range(self.num_constraints):
                new_row = []
                new_rhs = self._ensure_fraction(self.b[i])
                for j in range(self.num_vars):
                    coeff = self._ensure_fraction(self.A[i][j])
                    new_row.append(coeff)
                    if self.lower_bounds[j] != 0:
                        new_rhs -= coeff * self.lower_bounds[j]
                        transformed = True
                new_A.append(new_row)
                new_b.append(new_rhs)
            
            # Update bounds: Original L <= x <= U becomes 0 <= x' <= U - L
            for j in range(self.num_vars):
                if self.lower_bounds[j] != 0:
                    if self.upper_bounds[j] is not None:
                        new_upper = self._ensure_fraction(self.upper_bounds[j]) - self.lower_bounds[j]
                        new_bounds.append((Fraction(0, 1), new_upper))
                    else:
                        new_bounds.append((Fraction(0, 1), None))
                    transformed = True
                else:
                    new_bounds.append((Fraction(0, 1), self.upper_bounds[j]))
            
            self.A = new_A
            self.b = new_b
            self.bounds = new_bounds
            
            # Display transformed constraints
            if transformed and self.Print_Debug >= 1:
                print("\nTransformed constraints after lower bound transformation:")
                for i in range(self.num_constraints):
                    constr_parts = []
                    for j, a in enumerate(self.A[i]):
                        if a != 0:
                            sign = " + " if a > 0 and constr_parts else ""
                            constr_parts.append(f"{sign}{a} x{j+1}'")
                    print(f"  {''.join(constr_parts) if constr_parts else '0'} {self.rel[i]} {self.b[i]}")
                print("\nTransformed variable bounds:")
                for j, (lo, hi) in enumerate(self.bounds):
                    lb = str(lo) if lo is not None else "-Inf"
                    ub = str(hi) if hi is not None else "+Inf"
                    print(f"  x{j+1}': {lb} <= x{j+1}' <= {ub}")
            elif self.Print_Debug >= 1:
                print("\nNo lower bound transformation applied (all lower bounds are zero)")
    
        # 2. Check for conflicting bounds
        for j, (lo, hi) in enumerate(self.bounds):
            if lo is not None and hi is not None:
                try:
                    lo_frac = self._ensure_fraction(lo)
                    hi_frac = self._ensure_fraction(hi)
                    if lo_frac > hi_frac:
                        return 'infeasible'
                except:
                    continue
    
        # 3. Normalize equality constraints with negative RHS
        for i in range(self.num_constraints):
            if self.rel[i] == '=':
                try:
                    rhs = self._ensure_fraction(self.b[i])
                    if rhs < 0:
                        for j in range(self.num_vars):
                            self.A[i][j] *= -1
                        self.b[i] *= -1
                        self.eq_flip_flags[i] = True
                        if self.Print_Debug >= 2:
                            print(f"Flipped constraint {i+1} due to negative RHS: {rhs}")
                except:
                    continue
    
        # 4. Convert bounds to constraints if requested
        if convert_upper_bounds or convert_all_bounds:
            new_constraints = []
            
            for j in range(self.num_vars):
                lo, hi = self.bounds[j]
                
                if convert_all_bounds and lo is not None and self.CP_LB_treatment == 1:
                    try:
                        lo_frac = self._ensure_fraction(lo)
                        if lo_frac != 0:
                            new_row = [Fraction(0, 1) for _ in range(self.num_vars)]
                            new_row[j] = Fraction(1, 1)
                            new_constraints.append({
                                'row': new_row,
                                'rhs': lo_frac,
                                'rel': '>='
                            })
                    except:
                        continue
    
                if hi is not None:
                    try:
                        hi_frac = self._ensure_fraction(hi)
                        new_row = [Fraction(0, 1) for _ in range(self.num_vars)]
                        new_row[j] = Fraction(1, 1)
                        new_constraints.append({
                            'row': new_row,
                            'rhs': hi_frac,
                            'rel': '<='
                        })
                    except:
                        continue
    
            if new_constraints:
                self.eq_flip_flags.extend([False] * len(new_constraints))
                for con in new_constraints:
                    self.A.append(con['row'])
                    self.b.append(con['rhs'])
                    self.rel.append(con['rel'])
                self.num_constraints = len(self.A)
    
        return 'continue'

    def _validate_solution(self, solution):
        """Verify that the solution satisfies all constraints and bounds of the original problem.
        Args:
            solution: List of proposed variable values.
        Returns:
            bool: True if solution is feasible, False otherwise.
        """
        feasible = True
        
        if self.Print_Debug >= 1:
            print("\n=== Verifying Solution Feasibility Against Original Problem ===")
    
        # Check variable bounds against original bounds
        for j in range(self.original_num_vars):
            val = self._ensure_fraction(solution[j])
            lo, hi = self.original_bounds[j] if j < len(self.original_bounds) else (None, None)
            
            if lo is not None:
                lo_frac = self._ensure_fraction(lo)
                if val < lo_frac - Fraction(1, 1000000):
                    if self.Print_Debug >= 1:
                        print(f"Variable x{j+1} = {val} violates original lower bound {lo_frac}")
                    feasible = False
            if hi is not None:
                hi_frac = self._ensure_fraction(hi)
                if val > hi_frac + Fraction(1, 1000000):
                    if self.Print_Debug >= 1:
                        print(f"Variable x{j+1} = {val} violates original upper bound {hi_frac}")
                    feasible = False
    
        # Check constraints against original problem data
        for i in range(len(self.original_A)):
            lhs = Fraction(0, 1)
            for j in range(self.original_num_vars):
                coeff = self._ensure_fraction(self.original_A[i][j])
                val = self._ensure_fraction(solution[j])
                lhs += coeff * val
            
            rhs = self._ensure_fraction(self.original_b[i])
            rel = self.original_rel[i] if i < len(self.original_rel) else '<='
            
            tolerance = Fraction(1, 1000000)
            violation = False
            if rel == '<=' and lhs > rhs + tolerance:
                violation = True
            elif rel == '>=' and lhs < rhs - tolerance:
                violation = True
            elif rel == '=' and abs(lhs - rhs) > tolerance:
                violation = True
            
            if violation:
                if self.Print_Debug >= 1:
                    print(f"Constraint {i+1} violated: {lhs} {rel} {rhs}")
                feasible = False
    
        if self.Print_Debug >= 1:
            print(f"Solution is {'feasible' if feasible else 'infeasible'} against original problem")
        
        return feasible


    def solve(self):
        """Solve the LP problem using a two-phase simplex method with exact fractions."""
        try:
            print(f"\nCP_UB_as_LE setting: {self.CP_UB_as_LE}")
            print(f"CP_LB_treatment setting: {self.CP_LB_treatment}")

            # 1. Presolve: Check feasibility and transform problem
            presolve_status = self.Presolve(
                convert_upper_bounds=(self.CP_UB_as_LE == 1),
                convert_all_bounds=(self.CP_UB_as_LE == 2)
            )
            if presolve_status == 'infeasible':
                return None, None, 'infeasible'

            # 2. Phase 1: Find feasible solution
            phase1_c, phase1_A, phase1_b, phase1_basis = self.create_phase1_problem()
            
            if self.Print_Debug >= 1:
                print("\n=== Starting Phase 1 ===")
                print(f"Phase 1 Objective: {phase1_c}")
                basis_names = [self.var_names[idx] for idx in phase1_basis if idx < len(self.var_names)]
                print(f"Initial Basis: {basis_names}")

            phase1_tableau, phase1_basis = self.simplex_kernel(
                phase1_c, phase1_A, phase1_b, phase1_basis, sense='min', phase=1)

            if self.Print_Debug >= 1:
                print("\n=== Phase 1 Completed Successfully ===")
                basis_names = [self.var_names[idx] for idx in phase1_basis if idx < len(self.var_names)]
                print(f"Feasible Basis: {basis_names}")
                
                print("Basic variables:")
                for i, var_idx in enumerate(phase1_basis):
                    if var_idx < len(self.var_names):
                        var_name = self.var_names[var_idx]
                        var_value = phase1_tableau[i][-1]
                        print(f"    {var_name} = {var_value}")
                
                print("Non-basic variables:")
                for j in range(len(phase1_c)-1):
                    if j not in phase1_basis and j < len(self.var_names):
                        var_name = self.var_names[j]
                        if self.var_types[j] in ('slack', 'surplus'):
                            var_value = Fraction(0, 1)
                        else:
                            lo, hi = self.bounds[j] if j < len(self.bounds) else (None, None)
                            if lo is not None:
                                var_value = lo
                            elif hi is not None:
                                var_value = hi
                            else:
                                var_value = Fraction(0, 1)
                        print(f"    {var_name} = {var_value}")

            # =============================================
            # Improved Basis Cleaning Step after Phase 1
            # =============================================
            if self.Print_Debug >= 1:
                print("\n=== Cleaning Basis ===")
            
            cleaned_basis = phase1_basis.copy()
            cleaned_tableau = [row.copy() for row in phase1_tableau]
            artificial_in_basis = [
                i for i, var_idx in enumerate(cleaned_basis)
                if var_idx < len(self.var_types) and self.var_types[var_idx] == 'artificial'
            ]
            
            if artificial_in_basis:
                if self.Print_Debug >= 2:
                    print("Found artificial variables in basis, attempting to remove...")
                
                max_attempts = len(cleaned_tableau[0]) * len(artificial_in_basis) * 2
                attempt = 0
                while artificial_in_basis and attempt < max_attempts:
                    attempt += 1
                    row_idx = artificial_in_basis[0]
                    var_idx = cleaned_basis[row_idx]
                    if self.Print_Debug >= 3:
                        print(f"  Attempt {attempt}: Processing {self.var_names[var_idx]} in row {row_idx}")
                    
                    # Find all possible entering variables
                    candidates = []
                    for j in range(len(cleaned_tableau[0])-1):
                        if (j not in cleaned_basis and 
                            cleaned_tableau[row_idx][j] != 0 and 
                            self.var_types[j] != 'artificial'):
                            # Check if pivot maintains feasibility
                            pivot_val = cleaned_tableau[row_idx][j]
                            min_ratio = None
                            valid_pivot = True
                            for i in range(len(cleaned_tableau)-1):
                                if i != row_idx and cleaned_tableau[i][j] != 0:
                                    ratio = cleaned_tableau[i][-1] / cleaned_tableau[i][j]
                                    if cleaned_tableau[i][j] > 0:
                                        if min_ratio is None or ratio < min_ratio:
                                            min_ratio = ratio
                                    elif ratio <= 0:
                                        valid_pivot = False
                                        break
                            if valid_pivot:
                                candidates.append((j, abs(cleaned_tableau[row_idx][j])))
                    
                    # Sort candidates: prefer decision variables, larger coefficients
                    candidates.sort(key=lambda x: (
                        0 if self.var_types[x[0]] == 'decision' else 1,
                        -x[1]  # Prefer larger coefficients for numerical stability
                    ))
                    
                    pivot_success = False
                    for entering_col, _ in candidates:
                        if self.Print_Debug >= 3:
                            print(f"    Testing pivot with {self.var_names[entering_col]}")
                        
                        test_tableau = [row.copy() for row in cleaned_tableau]
                        test_basis = cleaned_basis.copy()
                        pivot_val = test_tableau[row_idx][entering_col]
                        
                        for k in range(len(test_tableau[row_idx])):
                            test_tableau[row_idx][k] /= pivot_val
                        
                        for k in range(len(test_tableau)):
                            if k != row_idx:
                                factor = test_tableau[k][entering_col]
                                for l in range(len(test_tableau[k])):
                                    test_tableau[k][l] -= factor * test_tableau[row_idx][l]
                        
                        test_basis[row_idx] = entering_col
                        
                        valid = True
                        for i in range(len(test_tableau)-1):
                            if test_tableau[i][-1] < -Fraction(1, 1000000):
                                valid = False
                                break
                        
                        if valid:
                            if self.Print_Debug >= 2:
                                print(f"    Successful pivot: {self.var_names[var_idx]} -> {self.var_names[entering_col]}")
                            cleaned_tableau = test_tableau
                            cleaned_basis = test_basis
                            pivot_success = True
                            break
                        elif self.Print_Debug >= 3:
                            print(f"    Pivot failed (infeasible RHS)")
                    
                    if pivot_success:
                        artificial_in_basis = [
                            i for i, var_idx in enumerate(cleaned_basis)
                            if var_idx < len(self.var_types) and self.var_types[var_idx] == 'artificial'
                        ]
                    else:
                        artificial_in_basis.pop(0)
                        if self.Print_Debug >= 3:
                            print(f"    No valid pivot found for {self.var_names[var_idx]}, moving to next")
                
                if artificial_in_basis:
                    all_zero = True
                    for row_idx in artificial_in_basis:
                        if abs(cleaned_tableau[row_idx][-1]) > Fraction(1, 1000000):
                            all_zero = False
                            break
                    
                    if all_zero:
                        if self.Print_Debug >= 1:
                            print("Note: Remaining artificial variables have zero values")
                    else:
                        if self.Print_Debug >= 1:
                            print("Failed to remove all artificial variables with non-zero values")
                        return None, None, 'infeasible'
            
            # Validate basis feasibility
            if self.Print_Debug >= 2:
                print("Validating cleaned basis feasibility...")
            for i in range(len(cleaned_tableau)-1):
                if cleaned_tableau[i][-1] < -Fraction(1, 1000000):
                    if self.Print_Debug >= 1:
                        print(f"Cleaned basis infeasible: negative RHS in row {i+1}: {cleaned_tableau[i][-1]}")
                    return None, None, 'infeasible'
            
            if self.Print_Debug >= 1:
                print("Basis after cleaning:")
                for i, var_idx in enumerate(cleaned_basis):
                    if var_idx < len(self.var_names):
                        print(f"  {self.var_names[var_idx]} = {cleaned_tableau[i][-1]}")

            # 3. Phase 2: Optimize original objective
            phase2_c, phase2_A, phase2_b, phase2_basis = self.create_phase2_problem(
                cleaned_tableau, cleaned_basis)

            if self.Print_Debug >= 1:
                print("\n=== Starting Phase 2 ===")
                print(f"Phase 2 Objective: {phase2_c}")
                print("Phase 2 Constraints:")
                for i in range(len(phase2_A)):
                    print(f"    {phase2_A[i]} {self.rel[i] if i < len(self.rel) else '='} {phase2_b[i]}")
                basis_names = [self.var_names[idx] for idx in phase2_basis if idx < len(self.var_names)]
                print(f"Initial Basis: {basis_names}")

            phase2_tableau, phase2_basis = self.simplex_kernel(
                phase2_c, phase2_A, phase2_b, phase2_basis,
                sense='max' if self.sense == 'max' else 'min', phase=2)

            # 4. Post-Solve: Extract solution
            solution = [Fraction(0, 1) for _ in range(self.original_num_vars)]
            
            # Extract basic variables
            for i, var in enumerate(phase2_basis):
                if var < self.original_num_vars:
                    solution[var] = phase2_tableau[i][-1]
                    lo, hi = self.bounds[var] if var < len(self.bounds) else (None, None)
                    if lo is not None and solution[var] < lo:
                        solution[var] = lo
                    if hi is not None and solution[var] > hi:
                        solution[var] = hi
            
            # Handle non-basic variables
            for j in range(self.original_num_vars):
                if j not in phase2_basis:
                    lo, hi = self.bounds[j] if j < len(self.bounds) else (None, None)
                    if lo is not None:
                        solution[j] = lo
                    elif hi is not None:
                        solution[j] = hi
                    else:
                        solution[j] = Fraction(0, 1)

            # Transform back to original variables
            if self.CP_LB_treatment == 2:
                for j in range(self.original_num_vars):
                    solution[j] += self.lower_bounds[j]

            # 5. Validate solution against original problem
            if not self._validate_solution(solution):
                if self.Print_Debug >= 0:
                    print("Solution is infeasible against original problem constraints")
                return None, None, 'infeasible'

            # Calculate objective value
            obj_value = Fraction(0, 1)
            for j in range(self.original_num_vars):
                obj_value += self._ensure_fraction(self.original_c[j]) * solution[j]

            if self.Print_Debug >= 0:
                print("\n=== Optimal Solution ===")
                for i, val in enumerate(solution):
                    print(f"x{i+1} = {val}")
                print(f"Objective value: {obj_value}")

            return solution, obj_value, 'optimal'

        except ValueError as e:
            if "unbounded" in str(e).lower():
                return None, None, 'unbounded'
            return None, None, 'infeasible'
        except Exception as e:
            if self.Print_Debug >= 0:
                print(f"\nError during solution: {str(e)}")
            return None, None, 'infeasible'

    def print_problem(self):
        """Print the problem formulation clearly showing both original and transformed forms."""
        print("\n" + "="*60)
        print("LINEAR PROGRAMMING PROBLEM")
        print("="*60)
        print(f"Variables: {self.original_num_vars}, Constraints: {self.num_constraints}\n")
        
        print("Original variable bounds:")
        for i, (lo, hi) in enumerate(self.original_bounds):
            lb = str(lo) if lo is not None else "-Inf"
            ub = str(hi) if hi is not None else "+Inf"
            print(f"  x{i+1}: {lb} <= x{i+1} <= {ub}")
        
        if self.CP_LB_treatment == 2:
            transformed_vars = []
            for i, (lo, hi) in enumerate(self.bounds):
                if self.lower_bounds[i] != 0:
                    lb = str(lo) if lo is not None else "-Inf"
                    ub = str(hi) if hi is not None else "+Inf"
                    transformed_vars.append(f"  x{i+1}': {lb} <= x{i+1}' <= {ub} (L={self.lower_bounds[i]})")
            
            if transformed_vars:
                print("\nTransformed variable bounds (x = x' + L):")
                print("\n".join(transformed_vars))
        
        if self.CP_UB_as_LE == 1:
            ub_constraints = []
            has_upper_bounds = False
            for i, (lo, hi) in enumerate(self.original_bounds):
                if hi is not None:
                    ub_constraints.append(f"  x{i+1} <= {hi}")
                    has_upper_bounds = True
            
            if ub_constraints:
                print("\nUpper bounds converted to constraints:")
                print("\n".join(ub_constraints))
            if has_upper_bounds:
                print("\nNote: Upper bounds are being treated as constraints (CP_UB_as_LE=1)")
        elif self.CP_UB_as_LE == 2:
            all_constraints = []
            has_bounds = False
            for i, (lo, hi) in enumerate(self.original_bounds):
                if lo is not None and lo != 0 and self.CP_LB_treatment == 1:
                    all_constraints.append(f"  x{i+1} >= {lo}")
                    has_bounds = True
                if hi is not None:
                    all_constraints.append(f"  x{i+1} <= {hi}")
                    has_bounds = True
            
            if all_constraints:
                print("\nBounds converted to constraints:")
                print("\n".join(all_constraints))
            if has_bounds:
                print("\nNote: Non-zero bounds are being treated as constraints (CP_UB_as_LE=2)")

        if self.CP_LB_treatment == 1 and self.CP_UB_as_LE != 2:
            lb_constraints = []
            has_non_zero_lower_bounds = False
            for i, (lo, hi) in enumerate(self.original_bounds):
                if lo is not None and lo != 0:
                    try:
                        lo_frac = self._ensure_fraction(lo)
                        if lo_frac != 0:
                            lb_constraints.append(f"  x{i+1} >= {lo}")
                            has_non_zero_lower_bounds = True
                    except:
                        continue
            
            if has_non_zero_lower_bounds:
                print("\nLower bounds converted to constraints (non-zero only):")
                print("\n".join(lb_constraints))
                print("\nNote: Non-zero lower bounds are being treated as constraints (CP_LB_treatment=1)")

        print(f"\n{'Maximize' if self.sense == 'max' else 'Minimize'}:")
        obj_parts = []
        for i, c in enumerate(self.original_c):
            if c != 0:
                sign = " + " if c > 0 and obj_parts else ""
                obj_parts.append(f"{sign}{c} x{i+1}")
        print("".join(obj_parts) if obj_parts else "0")
        
        print("\nSubject to:")
        for i in range(len(self.A)):
            constr_parts = []
            for j, a in enumerate(self.A[i]):
                if a != 0:
                    sign = " + " if a > 0 and constr_parts else ""
                    constr_parts.append(f"{sign}{a} x{j+1}")
            print(f"{''.join(constr_parts) if constr_parts else '0'} {self.rel[i]} {self.b[i]}")
        print("="*60 + "\n")

