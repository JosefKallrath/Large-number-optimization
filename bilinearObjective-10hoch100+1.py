"""

Section 5.3. Example 3: 
        Bilinear objective function and linear and quadratic constraints (MIQP)


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
	
  


import math
from mpmath import mp

def optimize_xy(N, M):
    """
    Solves max z = xy subject to:
    - x + y <= N
    - x^2 + y^2 <= M
    where x is integer and y is real
    """
    # Set precision
    mp.dps = 100
    
    N = mp.mpf(N)
    M = mp.mpf(M)
    
    x_max = min(math.floor(N), math.floor(mp.sqrt(M)))
    max_z = mp.mpf(0)
    best_x = 0
    best_y = mp.mpf(0)
    
    # Check boundary candidates
    for x in [0, x_max]:
        y = min(N - x, mp.sqrt(M - x*x))
        z = x * y
        if z > max_z:
            max_z = z
            best_x = x
            best_y = y
    
    # Check theoretical optimum points
    x_candidates = [
        math.floor(N/2),
        math.ceil(N/2),
        math.floor(mp.sqrt(M/2)),
        math.ceil(mp.sqrt(M/2))
    ]
    
    for x in x_candidates:
        if x < 0 or x > x_max:
            continue
        y = min(N - x, mp.sqrt(M - x*x))
        z = x * y
        if z > max_z:
            max_z = z
            best_x = x
            best_y = y
    
    # Full enumeration if needed
    if x_max <= 1000:
        for x in range(0, x_max + 1):
            y = min(N - x, mp.sqrt(M - x*x))
            z = x * y
            if z > max_z:
                max_z = z
                best_x = x
                best_y = y
    
    return int(best_x), best_y, max_z

def main():
    # Test with 100-digit numbers
    N = "100000000000000000000000000000000000000000000000001"
    M = "20000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001"
    
    x, y, z = optimize_xy(N, M)
    
    print(f"Optimal x: {x}")
    print(f"Optimal y: {y}")
    print(f"Maximum z: {z}")

if __name__ == "__main__":
    main()
