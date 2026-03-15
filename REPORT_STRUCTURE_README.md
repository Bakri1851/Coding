# Spatial Fish Population Modelling Under Harvesting Pressure

From Logistic ODEs to 2D Reaction-Diffusion and Sensitivity Analysis

## Overview

This report studies fish population dynamics under harvesting pressure using a progression of mathematical models. The work begins with time-only logistic ODE models, extends to 1D reaction-diffusion PDEs with spatially varying harvest policies, then introduces two-species competition and a three-zone maritime policy variant, before moving to 2D ocean simulations. A final sensitivity analysis tests how robust the main conclusions are to parameter changes.

The report is structured so that each modelling layer adds one new ingredient:

- baseline population growth
- harvesting
- spatial diffusion
- policy zoning
- species competition
- 2D spatial structure
- parameter sensitivity and robustness

## Research Questions

- How does harvesting alter long-term fish population behaviour in non-spatial and spatial models?
- How does diffusion affect biomass redistribution and spillover across management boundaries?
- How do zoning policies such as EEZ-style harvesting influence biomass, catch, and long-run equilibrium?
- How does species competition change the response to harvesting and spatial management?
- Which model parameters most strongly affect long-run outcomes?

## Report Structure

### 1. Abstract

Briefly summarise:

- the fisheries management problem
- the modelling progression
- the numerical methods used
- the main ecological and policy findings

### 2. Introduction

Include:

- motivation for modelling fisheries under harvesting pressure
- why spatial structure matters
- why policy boundaries such as EEZs are relevant
- the role of competition between species
- the report aims and research questions

### 3. Common Framework and Methods

Define the shared modelling framework once so later chapters stay focused on results.

Include:

- notation and variables
- dimensional and dimensionless forms
- parameter definitions
- harvesting methodology
- diffusion methodology
- boundary conditions
- numerical methods and solver choices
- stability conditions
- implementation and reproducibility overview

### 4. Baseline Non-Spatial Models

Purpose: establish the simplest dynamics and validate the analytical foundations.

#### 4.1 Logistic ODE

Include:

- governing equation
- nondimensional form
- analytic solution
- equilibria and interpretation
- numerical validation against analytic solution

#### 4.2 Logistic ODE with Harvesting

Include:

- harvested logistic equation
- harvest threshold behaviour
- sustainable vs overfished regimes
- MSY and MEY interpretation
- trajectory comparisons for different harvest rates

### 5. 1D Spatial Single-Species Models

Purpose: introduce space and show how diffusion and zoning change the dynamics.

#### 5.1 Pure Diffusion Validation

Include:

- diffusion-only PDE
- no-flux boundaries
- mass conservation check
- spreading of an initial Gaussian profile

#### 5.2 Logistic Reaction-Diffusion

Include:

- growth plus diffusion PDE
- comparison with uniform-state logistic behaviour
- interpretation of spatial smoothing

#### 5.3 EEZ Harvesting Policy

Include:

- piecewise harvest profile
- inside/outside EEZ comparison
- biomass and catch results
- space-time heatmaps
- spillover interpretation

#### 5.4 Stochastic Annual Harvest Schedules

Include:

- annual harvest-rate sampling setup
- shared schedule methodology
- robustness of results under variable policy intensity

### 6. 1D Competition and Policy Extensions

Purpose: show how species interaction and more realistic zoning modify the management picture.

#### 6.1 Two-Species Competition Model

Include:

- coupled Lotka-Volterra reaction-diffusion equations
- coexistence and exclusion conditions
- species biomass and catch comparisons
- spatial competition patterns

#### 6.2 Three-Zone Maritime Policy Variant

Include:

- territorial, EEZ, and international zones
- zone-specific harvesting assumptions
- diffusive spillover into heavily fished zones
- attractor robustness under different initial placements

### 7. 2D Ocean Extension

Purpose: test whether the main 1D findings remain meaningful in a more realistic spatial setting.

#### 7.1 Single-Species 2D Model

Include:

- 2D reaction-diffusion equation
- offshore no-flux and alongshore periodic boundaries
- 2D heatmaps and biomass summaries
- comparison of policy scenarios

#### 7.2 Two-Species 2D Model

Include:

- coupled 2D competition system
- coexistence vs exclusion scenarios
- spatial overlap and segregation patterns
- implications for management under multi-species dynamics

### 8. Sensitivity and Robustness Analysis

Purpose: identify which parameters most strongly drive the conclusions.

Include:

- one-at-a-time sensitivity analysis
- tornado plot
- top-parameter time-series comparisons
- fixed-time response curves
- single-parameter sweeps
- coexistence phase plane
- `(r1, H1)` heatmap
- outside-harvest policy sweep

End this chapter by stating which conclusions are robust and which depend strongly on parameter choice.

### 9. Discussion

Use this chapter to synthesize the whole report.

Include:

- what remains consistent from ODE to 2D
- how harvesting thresholds change with added spatial realism
- how diffusion creates spillover across policy boundaries
- how competition changes policy outcomes
- biological limitations of the model
- numerical and methodological limitations
- possible improvements or extensions

### 10. Conclusion

Answer the research questions directly.

Include:

- the main modelling findings
- the strongest policy takeaway
- what the progression from ODE to 2D reveals
- the next logical extension of the work

### 11. Appendices

Place supporting material here so the main report stays readable.

Suggested appendix content:

- longer derivations
- full parameter tables
- supplementary plots
- additional sweep results
- raw sensitivity tables
- implementation details

## Recommended Figure Strategy

Keep only the most important figures in the main body:

- model progression diagram
- shared parameter table
- ODE validation plot
- diffusion mass-conservation check
- EEZ biomass and catch comparison
- two-species coexistence/exclusion comparison
- key 2D heatmaps
- tornado plot
- coexistence phase plane
- outside-harvest policy sweep

Move repetitive snapshot panels and extra robustness figures to appendices.

## Suggested Writing Pattern for Each Results Chapter

For consistency, use the same structure in each technical chapter:

1. State the governing equations.
2. Define the scenario and parameters.
3. Show validation or benchmark behaviour.
4. Present the main figures and tables.
5. Interpret the result in ecological or policy terms.
6. End with a short chapter takeaway.

## Assumptions Behind This Structure

- The report is written for an academic audience at master's level.
- The full modelling pipeline is included, not just the core notebooks.
- Sensitivity analysis is treated as a main chapter, not supplementary material.
- The three-zone policy model is a subsection within the 1D policy chapter rather than a separate standalone chapter.
- Shared methods are grouped early to avoid repeating definitions in every chapter.
