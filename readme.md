## (Dis)Ordered Fractions

Uses AlphaFold's pLDDT score to calculate disordered (ordered) fractions of proteins.

Residues are labelled as disordered (ordered) if a given number of *consecutive* residues (threshold provided) have a pLDDT score below (above) a given value (threshold provided) 

Test cases for AF_Matches.csv taken from llps_minus dataset where source sequence could be found within AlphaFold PDB sequence (either exact match or target sequence truncated wrt. AlphaFold PDB sequence)



### WalkthroughSinglePDB.ipynb

Walks through key functions used to calculate fractions from pdb and includes some test cases



### WalkthroughMatchedProteins.ipynb

Example of running fraction calculations for multiple proteins where sequence match information is known (e.g. output from https://github.com/jackent601/CheckAlphaFoldPDBSequences)

## Next Steps

- Write wrapper to calculate features from disordered/ordered fraction sub-setting