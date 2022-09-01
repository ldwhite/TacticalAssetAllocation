# Tactical Asset Allocation

A paper was published in the SSRN electronic journal by Wouter J. Keller and Hugo S. van Putten of Flex Capital in 2012, titled Generalized Momentum and Flexible Asset Allocation, in which they develop a Tactical Asset Allocation (TAA) strategy algorithmicly. From reading their methodology in this paper, I recreated their strategy in Python from the period of 2019 to 2022 and found that not only did the strategy significantly outperform the benchmark, it also survived relatively unscathed during the 2020 market crash. The most significant change that I make from the original paper is the use of Mutual Information (MI). While correlation is ubiquitous throughout finance, correlation defines only a linear relationship between two variables and while the correlation coefficient is easy to understand intuitively, it does not provide the best description of association between variables whose relationship is nonlinear.

# References

Keller, W. J. and van Putten, H. S. (2012) Generalized Momentum and Flexible Asset Allocation (FAA) An Heuristic Approach, SSRN
