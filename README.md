# Experiments for "Order-Sensitive Sequential Interventions on Ideal Lattices"

`./run_all.sh` regenerates everything used in Section 7 and Appendix F. Requirements: Python 3.10+,
numpy, pandas, scipy, matplotlib (scikit-learn for the real-data part). All seeds are fixed.

| Folder / script | What it does | Paper |
| --- | --- | --- |
| `synthetic/lattice.py` | posets, ideal lattices, DP with stopping, reference paths | — |
| `synthetic/e1_support.py` | support separation: IPW vs naive, RMSE of reference score and order effect vs order propensity q, coverage of the Bernstein bound | Fig. 3(a), Prop. 5.5 |
| `synthetic/e2_planning.py` | regret of sequence-sensitive, pessimistic, order-insensitive, endpoint-pooled, greedy planners vs order-effect scale, order bias, sample size | Fig. 3(b), Fig. 4 |
| `synthetic/e3_cube.py` | least-squares projection onto cube-consistent fields; chi^2 cube test of edge additivity | Fig. 3(c) |
| `synthetic/e4_scaling.py` | DP states vs C(H+w, w) and number of plans | Fig. 3(d) |
| `sepsis/run_sepsis.py` | Sepsis Cases hospital log: support of all 10 pairs of ER actions (S2), main diamond LacticAcid / IV Antibiotics with cross-fitted AIPW for three outcomes (S1), tie-window sensitivity (S3), additivity check (S4), held-out planning with doubly robust values (S5), specification sensitivity (S6) | Table 2, Tables 3-5 |
| `assistments/run_assistments.py` | ASSISTments 2009-2010 skill-builder subset: support of the two orders for all 18,473 (target; skill pair) families (A1); classes containing both orders, within-class order effects with Benjamini-Hochberg and a within-class permutation null (A2) | Section 7, Fig. 5, Table 6 |
| `assistments/run_semisynthetic.py` | E5: planted order effects on the real support of the ASSISTments families (outcomes permuted within classes, known effect added); pooled vs within-class estimators: size, power, coverage, error vs effective two-sided support | Section 7, Fig. 6, Table 7 |

Requirements for the real-data part additionally include scikit-learn. Neither real dataset is redistributed;
see `sepsis/data/README.txt` and `assistments/data/README.txt` (the ASSISTments terms of use forbid
redistribution and require public code, which this package provides).
