# Enhanced Genetic Algorithm Approach for Multi-Year Maintenance and Rehabilitation Optimization for Large-Scale Infrastructure Networks

This repository contains the datasets, source code, and models associated with the research paper titled "Enhanced Genetic Algorithm Approach for Multi-Year Maintenance and Rehabilitation Optimization for Large-Scale Infrastructure Networks". This work is focused on improving maintenance and rehabilitation strategies for large-scale infrastructure networks through advanced genetic algorithms.

## Overview

The research presented in this paper addresses the significant challenge of multi-year network maintenance and rehabilitation optimization, a critical aspect of infrastructure asset management. While genetic algorithms (GAs) have traditionally been employed for this purpose, their effectiveness diminishes substantially as network sizes increase. This is primarily due to the limitations of conventional crossover and mutation operations, which often disrupt the composition of promising solutions, leading to a reduced probability of achieving feasible results.

To overcome these limitations, our study introduces an enhanced GA that incorporates two key innovations: a novel crossover technique that treats annual plans as a cohesive block of genes, and a unique mutation technique that applies linear programming (LP) to refine annual plans under varying budget scenarios. These advancements help maintain the integrity of each year's plan during the evolutionary process, while significantly improving local search capabilities.

We demonstrate the effectiveness of our hybrid LP-GA approach through two practical case studies: one focusing on a small-scale sewer network flushing program, and the other on a larger scale involving 13,610 pavement segments. Results from these case studies indicate that our proposed algorithm not only achieves rapid convergence but also maintains a 100% rate of generating feasible solutions, reaching optimal or near-optimal outcomes efficiently.

This work provides a sophisticated algorithmic tool for the field of infrastructure asset management, paving the way for further innovations in the sector.
