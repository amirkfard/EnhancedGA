# Enhanced Genetic Algorithm for Multi-Year Maintenance and Rehabilitation Optimization in Large-Scale Infrastructure Networks

This repository contains the datasets, source code, and experimental results accompanying the peer-reviewed journal article:

**Fard, A. K., & Yuan, X. X. (2024).** *Multiyear Maintenance and Rehabilitation Optimization for Large-Scale Infrastructure Networks: An Enhanced Genetic Algorithm Approach*.  
*Journal of Infrastructure Systems*, 30(4), 04024028.  
[https://doi.org/10.1061/JITSE4.ISENG-2489](https://doi.org/10.1061/JITSE4.ISENG-2489)

---

## Overview

This research addresses the complex and computationally intensive problem of optimizing multi-year maintenance and rehabilitation (M&R) plans for large-scale infrastructure networks. The study introduces an **enhanced genetic algorithm (GA)** framework designed to overcome the limitations of conventional GA methods when applied to real-world networks with thousands of assets and strict budgetary constraints.

Traditional GAs tend to degrade solution feasibility or performance as problem dimensionality increases, primarily due to the disruptive nature of standard crossover and mutation operators. Our enhanced GA integrates two key innovations:

- **Structured crossover**: A block-based crossover operator that exchanges entire annual plans (gene blocks), preserving temporal and financial consistency across solutions.
- **LP-guided mutation**: A linear programming (LP) module embedded in the mutation process to intelligently adjust asset-level decisions while ensuring budget adherence and feasibility.

These enhancements result in superior convergence behavior, feasibility retention, and solution quality in large-scale scenarios.

---

## Case Studies

We validated the proposed LP-GA framework on two infrastructure management problems:

1. **Sewer flushing optimization** (small-scale): Demonstrates the method’s adaptability and feasibility preservation in constrained operational contexts.
2. **Pavement rehabilitation planning** (large-scale): Involves 13,610 road segments and illustrates scalability, efficiency, and optimality under a multi-year capital planning scenario.

In both cases, the algorithm consistently achieved 100% feasible solutions and demonstrated significant improvements in solution quality and computational performance.


---

## How to Cite

If you use this code or dataset in your research, please cite the following article:

> Fard, A. K., & Yuan, X. X. (2024).  
> *Multiyear Maintenance and Rehabilitation Optimization for Large-Scale Infrastructure Networks: An Enhanced Genetic Algorithm Approach*.  
> *Journal of Infrastructure Systems*, 30(4), 04024028.  
> [https://doi.org/10.1061/JITSE4.ISENG-2489](https://doi.org/10.1061/JITSE4.ISENG-2489)

