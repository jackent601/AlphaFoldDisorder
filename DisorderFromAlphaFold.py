import numpy as np
from Bio.PDB import PDBParser

# ================================================================================================================================
#   DISORDER (pLDDT) Fractions
# ================================================================================================================================

def getpLDDTsFromAlphaFoldPDBModel(pdb_model):
    """
    AlphaFold stores plDDT as B-factor, pLDDT calculated per residue to only need to query first atom, returns array of pLDDT values (per residue)
    """
    return np.array([next(r.get_atoms()).get_bfactor() for r in pdb_model.get_residues()])


def getpLDDTsSubSequenceFromAlphaFoldPDBModel(pdb_model, startRes=None, endRes=None):
    """
    Returns pLDDTs of a sub-sequence within AF model, note: python is zero indexed, residue numbers are not!
    """
    if endRes is None and startRes is None:
        # read whole sequence
        return getpLDDTsFromAlphaFoldPDBModel(pdb_model)
    else:
        # filter for subsequence
        # Check both bounds provided
        assert startRes is not None and endRes is not None, "Must provide BOTH of start res and end res if specified"
        # SubSequencing Checks
        assert startRes <= endRes, "Start Residue number sub-sequence is greater than end residue number!"
        # Check Residue length is not larger than AlphaFold sequence
        _AFSequenceLength = len(list(pdb_model.get_residues()))
        assert endRes - startRes <= _AFSequenceLength, "Residue Sub Selection larger than AlphaFold sequence!"
        # Check bounds within AlphaFold Sequence
        assert endRes <= _AFSequenceLength, "Residue Sub Selection outside of AlphaFold Sequence!"

        return getpLDDTsFromAlphaFoldPDBModel(pdb_model)[startRes-1:endRes]


def getConsecutivepLDDTFromThreshold(pLDDTs, pLDDTThreshold, aboveThreshold=False):
    """
    finds the length of all stretches of residues consecutively below (or above if flag set) a pLDDT threshold i.e. 'disordered' (or 'ordered') stretches
    returns both the list of indices where each stretch starts, and the length of each stretch
    pLDDTs should be an array of pLDDTs from AF PDB file
    """
    # Finds indices in pLDDT list that are below threshold
    if aboveThreshold:
        pLDDTIndices = np.argwhere(pLDDTs >= pLDDTThreshold)[:, 0]
    else:
        pLDDTIndices = np.argwhere(pLDDTs <= pLDDTThreshold)[:, 0]

    # Catch Case of No Disorder (within threshold)
    if len(pLDDTIndices) == 0:
        return None, None

    # Need to duplicate final index value for calculating the lengths below in while loop
    pLDDTIndices = np.append(pLDDTIndices, pLDDTIndices[-1])

    # Find where difference between adjacent indices is greater than 1
    # (This indicates the begining of a new stretch of consecutively low pLDDTs regions)
    # (Adds one to correct for shifting array)
    consecutivepLDDTIndices = np.where(
        pLDDTIndices[1:] - pLDDTIndices[:-1] > 1)[0] + 1

    # Prepends with zero to account for first length
    consecutivepLDDTIndices = np.insert(consecutivepLDDTIndices, 0, 0)

    # Calculate length of each consecutively low pLDDT sequence
    consecutiveLens = np.zeros(len(consecutivepLDDTIndices)).astype(np.int64)
    for i in range(len(consecutivepLDDTIndices)):
        _length = 1
        _idx = consecutivepLDDTIndices[i]
        while pLDDTIndices[_idx+1] == pLDDTIndices[_idx] + 1:
            _length += 1
            _idx += 1
        consecutiveLens[i] = _length
    return consecutivepLDDTIndices, consecutiveLens


def getConsecutiveDisorderedFrompLDDTs(pLDDTs, pLDDTDisorderThreshold):
    """
    getConsecutivepLDDTFromThreshold with above flag set to false
    """
    return getConsecutivepLDDTFromThreshold(pLDDTs, pLDDTDisorderThreshold, aboveThreshold=False)


def getConsecutiveOrderedFrompLDDTs(pLDDTs, pLDDTOrderThreshold):
    """
    getConsecutivepLDDTFromThreshold with above flag set to True
    """
    return getConsecutivepLDDTFromThreshold(pLDDTs, pLDDTOrderThreshold, aboveThreshold=True)


def getFractionFrompLDDTs(pLDDTs, pLDDTThreshold, numberConsectuivelyDisorderThreshold, aboveThreshold=False):
    """
    Uses getConsecutivepLDDTFromThreshold to get length of all stretches of residues consecutively above or below a pLDDT threshold (depending on aboveThreshold flag) 
    Then filters lengths for those above or equal to length threshold
    Returns the ordered 'fraction', along with raw lengths of ordered residues within stretches above threshold
    """
    orderedStretchesIndices, orderedStretchesLengths = getConsecutivepLDDTFromThreshold(
        pLDDTs, pLDDTThreshold, aboveThreshold=aboveThreshold)

    if orderedStretchesIndices is None or orderedStretchesIndices is None:
        return 0, None

    orderedLengthsFiltered = orderedStretchesLengths[orderedStretchesLengths >=
                                                     numberConsectuivelyDisorderThreshold]

    return sum(orderedLengthsFiltered)/len(pLDDTs), orderedLengthsFiltered


def getOrderedFractionFrompLDDTs(pLDDTs, pLDDTDisorderThreshold, numberConsectuivelyDisorderThreshold):
    """
    getFractionFrompLDDTs with above flag set to True to find ordered sequences
    """
    return getFractionFrompLDDTs(pLDDTs, pLDDTDisorderThreshold, numberConsectuivelyDisorderThreshold, True)


def getDisorderedFractionFrompLDDTs(pLDDTs, pLDDTDisorderThreshold, numberConsectuivelyDisorderThreshold):
    """
    getFractionFrompLDDTs with above flag set to False to find Disordered sequences
    """
    return getFractionFrompLDDTs(pLDDTs, pLDDTDisorderThreshold, numberConsectuivelyDisorderThreshold, False)

# ================================================================================================================================
#   Functions From PDB Paths
# ================================================================================================================================

def getDisorderedFractionFromPDB(PDBpath, pLDDTDisorderThreshold, numberConsectuivelyDisorderThreshold, startRes=None, endRes=None):
    """
    getDisorderedFractionFrompLDDTs but reads directly from a PDB path, including option to sub sequence
    """
    # Read PDB
    parser = PDBParser()
    structure = parser.get_structure('auto_read', PDBpath)

    # Get Model, AlphaFold PDBs only have 1
    pdb_model = structure[0]

    # Get pLDDTs
    pLDDTs = getpLDDTsSubSequenceFromAlphaFoldPDBModel(
        pdb_model, startRes=startRes, endRes=endRes)

    # Get 'disordered' fractions
    return getDisorderedFractionFrompLDDTs(pLDDTs, pLDDTDisorderThreshold, numberConsectuivelyDisorderThreshold)


def getOrderedFractionsFromPDB(PDBpath, pLDDTDisorderThreshold, pLDDTOrderThreshold, numberConsectuivelyDisorderThreshold, numberConsectuivelyOrderThreshold, startRes=None, endRes=None):
    """
    Reads directly from a PDB path, including option to sub sequence
    Gets ordered and disordered fractions
    """
    # Read PDB
    parser = PDBParser()
    structure = parser.get_structure('auto_read', PDBpath)

    # Get Model, AlphaFold PDBs only have 1
    pdb_model = structure[0]

    # Get pLDDTs
    pLDDTs = getpLDDTsSubSequenceFromAlphaFoldPDBModel(
        pdb_model, startRes=startRes, endRes=endRes)

    # Get 'disordered' fractions
    dFrac, dLengths = getDisorderedFractionFrompLDDTs(
        pLDDTs, pLDDTDisorderThreshold, numberConsectuivelyDisorderThreshold)

    # Get 'Ordered' fractions
    oFrac, oLengths = getOrderedFractionFrompLDDTs(
        pLDDTs, pLDDTOrderThreshold, numberConsectuivelyOrderThreshold)

    return dFrac, dLengths, oFrac, oLengths


def getOrderedFractionsFromPDB_Config(PDBpath, CONFIG, startRes=None, endRes=None):
    """
    see getOrderedFractionsFromPDB, identical but uses config dictionary to tidy code
    """
    # unpack CONFIG
    pLDDT_DisorderThreshold = CONFIG['pLDDT_DISORDER_THRESHOLD']
    pLDDT_OrderThreshold = CONFIG['pLDDT_ORDER_THRESHOLD']
    ConsecutiveDisorderThreshold = CONFIG['CONSECUTIVE_DISORDER_THRESHOLD']
    ConsecutiveOrderThreshold = CONFIG['CONSECUTIVE_ORDER_THRESHOLD']

    return getOrderedFractionsFromPDB(PDBpath,
                                      pLDDT_DisorderThreshold,
                                      pLDDT_OrderThreshold,
                                      ConsecutiveDisorderThreshold,
                                      ConsecutiveOrderThreshold,
                                      startRes=startRes,
                                      endRes=endRes)

# ================================================================================================================================
#   Functions From AlphaFold Matches DataFrames (See https://github.com/jackent601/CheckAlphaFoldPDBSequences)
# ================================================================================================================================
def addOrderFractionToDFofAFMatches(AFMatchDF, 
                                    pathToPDBRootDir, 
                                    pLDDT_DisorderThreshold, 
                                    pLDDT_OrderThreshold,
                                    ConsecutiveDisorderThreshold, 
                                    ConsecutiveOrderThreshold,
                                    debug=False):
    """
    pathToPDBRootDir: path to pdb root diectory    
    pLDDT_DisorderThreshold: Threshold to determine disordered residues (pLDDT below or equal to)
    pLDDT_OrderThreshold: Threshold to determine Ordered residues (pLDDT aobve or equal to)
    ConsecutiveDisorderThreshold: Length of consecutive disordered residues to determined to be disordered
    ConsecutiveOrderThreshold: Length of consecutive ordered residues to determined to be ordered
    
    AFMatchDF: Dataframe of AlphaFold Sequence match information 
        (see Check Alpha Fold Sequence Info at https://github.com/jackent601/CheckAlphaFoldPDBSequences)
    Dataframe must have the following features:
        PDB_path: local path to pdb file
        ExactMatch: Whether exact sequence match
        AFStartResidueOverlap: if not exact match which AF residue to read from
        AFEndResidueOverlap: if not exact match which AF residue to read to
    """
    # Initialise
    dFracs = []
    oFracs = []
    
    # Iterate DF
    for index, row in AFMatchDF.iterrows():
        # Extract DF Info
        full_pdb_path = os.path.join(pathToPDBRootDir, row['PDB_path'])
        ExactMatch = row['ExactMatch']
        _debug = f'{os.path.split(full_pdb_path)[1]}'

        # Get Sequence Read Info
        if ExactMatch:
            startResRead = None
            endResRead = None
            _debug += ' - Exact Match, '
        else:
            startResRead = row['AFStartResidueOverlap']
            endResRead = row['AFEndResidueOverlap']
            _debug += f' - Truncated Match, read from res {startResRead} to res {endResRead} '
        
        # Calculate Order Fractions
        dFrac, _, oFrac, _ = getOrderedFractionsFromPDB(full_pdb_path,
                                                                      pLDDT_DisorderThreshold,
                                                                      pLDDT_OrderThreshold,
                                                                      ConsecutiveDisorderThreshold,
                                                                      ConsecutiveOrderThreshold,
                                                                      startRes=startResRead,
                                                                      endRes=endResRead)
        # Add to lists
        dFracs.append(dFrac)
        oFracs.append(oFrac)
        
        # Debug
        _debug += f'\n\tDisordered Frac: {100*dFrac:0.1f}, Order Frac: {100*oFrac:0.1f}'
        if debug:
            print(_debug)
        
    # Add Fractions to DF
    AFMatchDF['DisorderedFrac'] = dFracs
    AFMatchDF['OrderedFrac'] = oFracs
    return AFMatchDF

def addOrderFractionToDFofAFMatches_Config(AFMatchDF, CONFIG, debug=False):
    """
    See addOrderFractionToDFofAFMatches, identical but uses config dictionary to tidy code
    """
    # Unpack CONFIG
    pathToPDBRootDir = CONFIG['PATH_TO_ROOT_PDB']
    pLDDT_DisorderThreshold = CONFIG['pLDDT_DISORDER_THRESHOLD'] 
    pLDDT_OrderThreshold = CONFIG['pLDDT_ORDER_THRESHOLD']
    ConsecutiveDisorderThreshold = CONFIG['CONSECUTIVE_DISORDER_THRESHOLD'] 
    ConsecutiveOrderThreshold = CONFIG['CONSECUTIVE_ORDER_THRESHOLD']
    
    return addOrderFractionToDFofAFMatches(AFMatchDF,
                                           pathToPDBRootDir,
                                           pLDDT_DisorderThreshold,
                                           pLDDT_OrderThreshold,
                                           ConsecutiveDisorderThreshold,
                                           ConsecutiveOrderThreshold,
                                           debug=debug)