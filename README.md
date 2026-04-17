# adaptive_cycle
## usage
adaptive_cycle.py [input file] --time [t] --save-prefix [prefix]
## Input file format

    n
    x1_0 x2_0 ... xn_0
    j1 j2 ... jn : alpha1 alpha2 ... alphan

    xi_0 is an initial state of agent i.
    Each line represents monomial x1^j1*...*xn^jn.
    Parametr alphai represent coefficient of that monomial in i-th equation of model.
    Note: it shall be simplified soon
 
## Directiories
inputs -- self explanatory directiory

[dir_name] -- output directiory
biomass.csv - numerical solution for given ODE

[parameter name].csv - specific parameter outputs

linear-graph.gif - animation for parameters based on linearisation. Thickness of edge represents relative intensity of interaction. Color represents positivity/negativity. Size of node represents relative biomass of corresponding species

plots.pdf - plots for parameters and biomass. (NOTE: soon it shall be divided to different plots for interaction methodologies)

hypergraph.gif - an attempt to visualize hypergraph-based interactions.

phase_curve.pdf - 3d plot visualising potential, connectedness and resilience for chosen parameters at chosen time window. TODO: option to choose parameters and time window during program execution, currently hard-coded in source file.

TODO: translate all labels to english
