![CatFish Logo](CatFish.png)

by [Petabyte Economics Corp](https://www.petabyteeconomics.com).

(c) Petabyte Economics Corp., Jan. 2024. All Rights Reserved. See [LICENSE.md](./LICENSE.md) for more information.

# About CatFish 

[CatFish](https://www.petabyteeconomics.com/catfish.html) is a Bayesian **cat**egorical variable multiplicative **Poisson** economic demand model with hierarchical priors. Its inference uses Gibbs, and optionally Metropolis-within-Gibbs, Markov chain Monte Carlo sampling. The model supports any number and specification of categorical variables across any number and type of dimensions. The code is written in MATLAB.

For more information, see the CatFish:
- [Webpage](https://www.petabyteeconomics.com/catfish.html)
- [Brochure](https://www.petabyteeconomics.com/files/CatFish-Brochure.pdf)
- [Handbook](https://www.petabyteeconomics.com/files/CatFish-Handbook.pdf)

# Classes 

The software consists of five classes:
- Model.m
- Dimension.m 
- Partition.m
- Block.m: 
- Family.m: 

A model object provides all of the methods and properties needed to assemble, make inferences from, and analyze a CatFish model. The model object uses all other classes, and objects from those classes are stored in the model object's arrays. However, the block objects perform the Gibbs sampling, and the family objects perform Metropolis sampling.

# Inference 

To run the model use a script with the following steps:

## Building a topology 

1. Instantiate a model object.
2. Define dimensions using vectors of labels.
3. Define partitions that assign labels (from one or more dimensions) to categories.
4. Create blocks of categorical variables from partitions' categories.[^1] 
5. Assign priors to categories within categorical variables using families.

[^1]: Most blocks use a single user-defined partition for one or two dimensions and automatically create degenerate partitions for other dimensions. 

## Inference 

6. Load the training data.
7. Do the in-sample inference.
8. Do any out-of-sample inference (draws from possibly hierarchical priors).
9. Save the results (thetas and alphabetas).
10. Optionally create lambda vectors for forecasting.

# Example Script

An example script - example1.m - is provided with the software.

This script infers parameters for a model of "Startup Cities." The model is not realistic (or even sensible); it is intended solely to provide examples of the software's functionality. The Startup Cities example infers the number of first "growth" venture capital financings of startups in 198 US towns and cities each year for three industries from 1981 to 2025. 

The model has:
- 3 dimensions (198 places x 3 industries x 45 years)
- 16 user-defined partitions (2 are degenerate)
- 13 blocks corresponding to 6 categorical variables, including:
-- 3 x 1-d global factors (blocks 1, 2, 3)
-- 1 x 2-d global interaction factor (block 4)
-- 1 x 2-d local factor (block 5)
-- 1 x 1-d scale factor (blocks 6-13)
- 5 families:
-- 3 x type 1 (Gibbs only)
-- 1 x type 2 (Metropolis only)
-- 1 x type 3 (Gibbs + Metropolis)

Required data files:
- sc_dim_place.txt - Dimension file for places, with place -> state partition.
- sc_dim_year.txt - Dimension file for years, used for its year -> decade partition.
- sc_anchorfund_cat.txt - Partition file, with anchorcat -> place, year
- sc_growthdeals.txt - Training data with growthdeals -> placestate,industry,year

Produced files:
- GibbsInferences.txt - Summary of results for the thetas.

# Contact Us

Please contact [info@petabyteeconomics.com](mailto:info@petabyteeconomics.com) for more information about CatFish.

(c) [Petabyte Economics Corp](https://www.petabyteeconomics.com), 2024.