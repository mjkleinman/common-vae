# TODO:

## Long-term
- multiview reconstruction like what Stefano said (for this I can use the third
  party blender (as Alessandro suggested)
- Think about how this can be applied in the "meta" or the multi-task setting

## Medium-term
- ddsprites action experiment where two images (with a shifted object are inputted, as well as the action (such as "--left 4")
- can the formulation be trivially generalized to capture the common
  information between different subsets? (Ans. Not trivially)

## short term

- Add in annealing schedule for betaU and betaC where the ratio is kept
  constant
- Common deformable template (Stefano's suggestion)
- Add option to set the common infomration for the ddsprites 
- Update visualizer to work with both views
- Include the KL values on the plots.

## Todo's from last meeting

1) Implement the metrics for disentanglement
2) Thorough literature review on disentanglement
3) Updated codebase w/ some refactoring
3.1) Update the visualizations to see both views
4) Run the experiments that Alessandro suggested [different viewpoints for the shapes, and dsprites with shift] [done -- Run the other]
5) Implement MINE-like technique for approximating the mutual information
7) Implement CelebA Paired
8) Reply to Stefano
9) TODO: run similar experiments on videos

# Done
- Added free bits
