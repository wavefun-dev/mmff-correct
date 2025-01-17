This illustrates the computation of relative energies at the ωB97X-V/6-311+G(2df,2p)[6-311G*] ("XV") level of theory using MMFF-optimized geometries

test.py:      This performas an analysis on molecules that were excluded from training the model.
              These molecules include the target data (energy) so we can see how our predictions perform.  See test.h5 for structure.

run_model.py: This can be used to apply the model on an input molecule, 'input.xyz'.  It will compute a relative energy offset 
              compared to another conformer of the same molecule. 
