import numpy as np
from Bio.PDB import PDBParser
import os

# ================================================================================================================================
#   (DIS)ORDER (pLDDT) REGIONS
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


def getpLDDTRegionsFromThreshold(pLDDTs, pLDDTThreshold, aboveThreshold=False):
    """
    finds the length of all stretches of residues consecutively below (or above if flag set) a pLDDT threshold i.e. 'disordered' (or 'ordered') stretches
    returns both the list of indices where each stretch starts, and the length of each stretch
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
    nonConsecutivepLDDTIndices = np.where(
        pLDDTIndices[1:] - pLDDTIndices[:-1] > 1)[0] + 1

    # Prepends with zero to account for first length
    nonConsecutivepLDDTIndices = np.insert(nonConsecutivepLDDTIndices, 0, 0)

    # Calculate length of each consecutively low pLDDT sequence
    consecutiveLens = np.zeros(len(nonConsecutivepLDDTIndices)).astype(np.int64)
    for i in range(len(nonConsecutivepLDDTIndices)):
        _length = 1
        _idx = nonConsecutivepLDDTIndices[i]
        while pLDDTIndices[_idx+1] == pLDDTIndices[_idx] + 1:
            _length += 1
            _idx += 1
        consecutiveLens[i] = _length
    return nonConsecutivepLDDTIndices, consecutiveLens

def getLengthFilteredpLDDTRegionsFromThreshold(pLDDT, pLDDTThreshold, lengthThreshold, aboveThreshold=False):
    """
    getpLDDTRegionsFromThreshold but also filters for only regions above a certain consecutive length
    """
    # Run normal (dis)order calculations to get pLDDT region start indices and lengths
    orderedStretchesIndices, orderedStretchesLengths = getpLDDTRegionsFromThreshold(pLDDTs=pLDDT,
    pLDDTThreshold=pLDDTThreshold, 
    aboveThreshold=aboveThreshold)

    # Filter for only those indices above the consecutive residue threshold
    orderedStretchesIndicesLengthFiltered = orderedStretchesIndices[orderedStretchesLengths>=lengthThreshold]
    orderedStretchesLengthsFiltered = orderedStretchesLengths[orderedStretchesLengths>=lengthThreshold]

    return orderedStretchesIndicesLengthFiltered, orderedStretchesLengthsFiltered

def getpLDDTRegionStartStopIndices(pLDDT, pLDDTThreshold, lengthThreshold, aboveThreshold=False):
    """
    Rather than return indice start and region length, gets actual indices of region start,stops. returning an array of arrays
    """ 
    # Get Region start indices and length (from thresholds)
    orderedStretchesIndicesLengthFiltered, orderedStretchesLengthsFiltered = getLengthFilteredpLDDTRegionsFromThreshold(pLDDT=pLDDT, 
    pLDDTThreshold=pLDDTThreshold, 
    lengthThreshold=lengthThreshold, 
    aboveThreshold=aboveThreshold)

    # get read-from/read-to indices for each identified region
    # python indexing is inclusive:exclusive, hence for region starting at index i_start, and length L, region is read by pLDDT[i_start:i_start+L] to read the region
    return [[start, start+length] for start, length in zip(orderedStretchesIndicesLengthFiltered, orderedStretchesLengthsFiltered)]

def getRegionStartStopResiduesFrompLDDTs(pLDDT, pLDDTThreshold, lengthThreshold, aboveThreshold=False, startRes=None):
    """
    Identical to getpLDDTRegionStartStopIndices, but returns RESIDUE NUMBERS (by adding 1 to indices), 
    option to include original offset (default is no offset, i.e. start residue = 1) compared to whole protein
    """
    # Get indices
    regionIndices = getpLDDTRegionStartStopIndices(pLDDT=pLDDT,
    pLDDTThreshold=pLDDTThreshold, 
    lengthThreshold=lengthThreshold,
    aboveThreshold=aboveThreshold)
    # Off set from initil subsetting 
    residueOffset = 1+(startRes-1) if startRes is not None else 1
    return [[regionPair[0]+residueOffset, regionPair[1]+residueOffset] for regionPair in regionIndices]

def processRegionFrompLDDTs(pdb_model, pLDDTThreshold, lengthThreshold, func, aboveThreshold=False, startRes=None, endRes=None, **kwargs):
    """
    Takes an arbitrary function with signature func(model, startResidue, stopResidue, *args) and calls it for each identified region within a pdb model
    Regions are identified from pLDDT scores based on thresholds
    """
    # First get pLDDTs
    pLDDTs = getpLDDTsSubSequenceFromAlphaFoldPDBModel(pdb_model, startRes=startRes, endRes=endRes)
    
    # Get Regions of interest
    regionResidueStartStopPairs = getRegionStartStopResiduesFrompLDDTs(pLDDTs, pLDDTThreshold, lengthThreshold, aboveThreshold=aboveThreshold, startRes=startRes)
    
    # Loop through each region calling functions
    results = []
    for regionResidueStartStop in regionResidueStartStopPairs:
        # Unpack Values
        _residueStart, _residueStop = regionResidueStartStop
        # Call Function
        results.append(func(pdb_model, _residueStart, _residueStop, **kwargs))
    
    return results

def getDisorderedRegionsFrompLDDTs(pLDDTs, pLDDTDisorderThreshold):
    """
    getConsecutivepLDDTFromThreshold with above flag set to false
    """
    return getpLDDTRegionsFromThreshold(pLDDTs, pLDDTDisorderThreshold, aboveThreshold=False)

def getOrderedRegionsFrompLDDTs(pLDDTs, pLDDTOrderThreshold):
    """
    getConsecutivepLDDTFromThreshold with above flag set to True
    """
    return getpLDDTRegionsFromThreshold(pLDDTs, pLDDTOrderThreshold, aboveThreshold=True)




# ================================================================================================================================
#   (DIS)ORDER (pLDDT) REGIONS
# ================================================================================================================================


# ================================================================================================================================
#   (DIS)ORDER (pLDDT) FRACTIONS
# ================================================================================================================================


def getFractionFrompLDDTs(pLDDTs, pLDDTThreshold, numberConsectuivelyDisorderThreshold, aboveThreshold=False):
    """
    Uses getConsecutivepLDDTFromThreshold to get length of all stretches of residues consecutively above or below a pLDDT threshold (depending on aboveThreshold flag) 
    Then filters lengths for those above or equal to length threshold
    Returns the ordered 'fraction', along with raw lengths of ordered residues within stretches above threshold
    """
    orderedStretchesIndices, orderedStretchesLengths = getpLDDTRegionsFromThreshold(
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
#   End-To-End Fraction Functions From PDB Paths
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
#   Functions For Data Frames with AlphaFold Sequence Matches Information, probabily should be in separate script 
#   See https://github.com/jackent601/CheckAlphaFoldPDBSequences for examples/generation
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