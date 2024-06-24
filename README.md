# Conserved Ising 2D

## Getting started
This research code is written in the Python and depends on numba (last tested with version '0.53.1').
The main file for the mean-field theory simulations is ``liquid_mcmc.jl``.
You can see options by running:

    python liquid_mcmc.py --help
    
The theoretical details of this code can be found in the reference below.

## Citations
If you found this code useful for your research, you can cite it using the below bibtex entry:

    @article{fall2023optimized,
	  title={An optimized species-conserving Monte Carlo method with potential applicability to high entropy alloys},
	  author={Fall, Aziz and Grasinger, Matthew and Dayal, Kaushik},
	  journal={Computational Materials Science},
	  volume={217},
	  pages={111886},
	  year={2023},
	  publisher={Elsevier}
    }
    
## Acknowledgements
Thanks to Aziz Fall, who developed the original method and implementation (https://github.com/azizfall/Local_Structure_MC).
