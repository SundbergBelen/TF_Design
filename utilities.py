import os
import pyrosetta
import inspect
from datetime import datetime

DEBUG = False

def set_debug(value: bool):
	global DEBUG
	DEBUG = value

def dprint(*args, **kwargs):
	if DEBUG:
		frame = inspect.currentframe().f_back
		func_name = frame.f_code.co_name
		print(f"[DEBUG][{func_name}]", *args, **kwargs)

def log(msg, level="INFO", thread=None, indent=0):
	frame = inspect.currentframe().f_back
	func_name = frame.f_code.co_name

	timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	thread_str = f"[ THREAD {thread} ]" if thread is not None else ""
	indent_str = "    " * indent  # 4 spaces per level

	print(f"[{timestamp}][{level}]{thread_str}[{func_name}] {indent_str}{msg}", flush=True)

def move_files(start_folder: os.path, destination_folder: os.path) -> None:
    """
    moves file from one folder to the other,
    @param start_folder: path to directory containing the files
    @param destination_folder: path to directory to move the files into.
    """
    for file in os.listdir(start_folder):
        start_path = os.path.join(start_folder, file)
        destinantion_path = os.path.join(destination_folder, file)
        os.replace(start_path, destinantion_path)

def get_mutable_list_from_residue_selector(
        residue_selector: pyrosetta.rosetta.core.select.residue_selector.ResidueSelector, pose: pyrosetta.Pose,
        chain: str) -> list:
    """
    Gives a list of residues assuming that the first residue of the chain is 1. Format is required for proteinmpnn.

    @param residue_selector: Residue_selector, residue selector from residues in a selected chain
    @param pose: pyrosetta.Pose
    @param chain: str, pdb chain index that contains residues in residue selector
    @return:
    """
    # Get a list of chain identifiers in the pose
    #chain_ids = [pose.chain(i) for i in range(1, pose.total_residue() + 1)]
    #print(chain_ids)

    # Use the first chain identifier as an example; replace with the desired logic
    #first_chain_id = chain_ids[0]
    #chain_start = pose.chain_begin(pyrosetta.rosetta.core.pose.get_chain_id_from_chain(first_chain_id, pose))
    chain_id = str(1)
    res_pose = residue_selector.apply(pose)
    mutable_residues = []
    
    chain_start = pose.chain_begin(pyrosetta.rosetta.core.pose.get_chain_id_from_chain("A", pose))
    for i, x in enumerate(res_pose):
        if x:
            mutable_residues.append(str(i + 1 - chain_start))
    return mutable_residues

def make_dir(dir_path: os.path) -> None:

    '''
    Creates a directory at the desired path, will fail if directory exists already:
    This will not create the containing directories.
    @rtype: object
    @param dir_path: path to directory that needs to be made
    @return: None
    Example Usage:
        make_dir("models_that_work")
        make_dir("models_that_work/ABC")
    '''

    try:
        os.makedirs(dir_path)
    except:
        log(f"Directory {dir_path} already exists")


def get_keys_with_lowest_scores(scores_dict, num):
    """
    Returns lowest scoring dict entries, based on index 2 of the value list of the dictionary. It is meant to be used
    with the dictionaries made by energy_methods.get_designs() or protein_mpnn_mutator.protein_mpnn_designs().

    @param scores_dict: dictionary with at least 3 elements, and where index 2 is the metric filtering.
    @param num: number of keys to pass, must be less than the size of scores_dict
    @return: list of keys for scores_dict
    """
    sorted_scores = sorted(scores_dict.items(), key=lambda x: x[1][2], reverse=True)
    keys = [item[0] for item in sorted_scores[:num]]
    return keys


def find_close_to_chain(target_chain: str, docked_pdb: os.path, distance: int = 7) \
        -> pyrosetta.rosetta.core.select.residue_selector.ResidueSelector:
    """
    Finds residues within a given distance from the selected chain. PDB Input must be already docked.
    @param target_chain: string to indicate chain based on PDB indexes
    @param docked_pdb:  path to PDB containing docked models
    @param distance: Neighborhood search distance, default is 7 A
    @return: Pyrosetta residue selector object <pyrosetta.rosetta.core.select.residue_selector>
    """

    hsa_pose: pyrosetta.Pose = pyrosetta.pose_from_pdb(docked_pdb)
    chainB = pyrosetta.rosetta.core.select.residue_selector.ChainSelector(
        pyrosetta.rosetta.core.pose.get_chain_id_from_chain(target_chain, hsa_pose))
    nearChainB = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector()
    nearChainB.set_focus_selector(chainB)
    nearChainB.set_distance(distance)
    nearChainB.set_include_focus_in_subset(False)

    return nearChainB


def fetch_binder(input_pdb: os.path) -> os.path:
    """
    Matches HSA model to mini-binder model.
    @param input_pdb: path to HSA model after Neel's step. Specific for HA outputs.
    @return: path to corresponding helical bundle - HA complex made during Kyle's step.
    """
    for file in (os.listdir("helical_bundles_hsa_complex")):
        target = file.split(("~"))[0]
        if target in input_pdb:
            x = file
            break
    return os.path.join("helical_bundles_hsa_complex", x)


def get_sequences_from_mpnn(path_to_seq: os.path) -> list:
    '''
    Gets sequences from proteinmpnn_glab_version output file.
    @param path_to_seq: Path to sequence file, output from proteinmpnn_glab_version
    @return: list of str sequences from proteinmpnn_glab_version output
    '''
    mpnn_seq = []
    with open(path_to_seq, 'r') as sequences:
        for line in sequences:
            if ">" in line:
                continue
            mpnn_seq.append(line.strip())
        mpnn_seq.pop(0)
    return mpnn_seq


def get_fa_file_name(pdb_path) -> str:
    """
    This is useful when you have already generated proteinmpnn_glab_version sequences and want to access them.
    @param pdb_path: path to pdb file
    @return: returns expected .fa file name from proteinmpnn_glab_version
    """
    return str(pdb_path.split("/")[-1])[0:-4] + ".fa"


def makeFaceFiles(pose: pyrosetta.Pose, faceA: pyrosetta.rosetta.core.select.residue_selector.ResidueSelector,
                  faceB: pyrosetta.rosetta.core.select.residue_selector.ResidueSelector, outputFaceA: os.path,
                  outputFaceB: os.path) -> None:
    """
    Generates correspoding face files for Interface Energy rosetta script, residue selector could be on the same chain.
    @param pose: Pyrosetta.Pose
    @param faceA: Residue selector of chain A
    @param faceB: residue selector of chain B
    @param outputFaceA: output file path for chain A
    @param outputFaceB: output file path for chain B
    @return: None
    """
    with open(outputFaceA, "w") as output:
        for index, boolValue in enumerate(faceA.apply(pose)):
            if boolValue:
                resInfo = pose.pdb_info().pose2pdb(index + 1)
                output.write(resInfo.split(" ")[1] + " " + resInfo.split(" ")[0] + " _\n")

    with open(outputFaceB, "w") as output:
        for index, boolValue in enumerate(faceB.apply(pose)):
            if boolValue:
                resInfo = pose.pdb_info().pose2pdb(index + 1)
                output.write(resInfo.split(" ")[1] + " " + resInfo.split(" ")[0] + " _\n")


def makeResidueSelectors(pose: pyrosetta.Pose, chain1: str, chain2: str, radius: float) -> [
    pyrosetta.rosetta.core.select.residue_selector.ResidueSelector,
    pyrosetta.rosetta.core.select.residue_selector.ResidueSelector]:
    """
    Generates two residue selectors corresponding to the interface between chain 1 and chain 2 of a given pose.
    @param pose: Pyrosetta.pose
    @param chain1: str, chain index from PDB to chain 1
    @param chain2: str, chain index from PDB to chain 2
    @param radius: float, allowed residues for interactions
    @return:  list with two residue selectors
    """
    chainA = pyrosetta.rosetta.core.select.residue_selector.ChainSelector(
        pyrosetta.rosetta.core.pose.get_chain_id_from_chain(chain1, pose))
    chainB = pyrosetta.rosetta.core.select.residue_selector.ChainSelector(
        pyrosetta.rosetta.core.pose.get_chain_id_from_chain(chain2, pose))

    nearChainA = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector()
    nearChainA.set_focus_selector(chainA)
    nearChainA.set_distance(radius)
    nearChainA.set_include_focus_in_subset(True)

    nearChainB = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector()
    nearChainB.set_focus_selector(chainB)
    nearChainB.set_distance(radius)
    nearChainB.set_include_focus_in_subset(True)
    interface = pyrosetta.rosetta.core.select.residue_selector.AndResidueSelector(selector1=nearChainA,
                                                                                  selector2=nearChainB)

    chainAList = pyrosetta.rosetta.utility.vector1_unsigned_long()
    chainBList = pyrosetta.rosetta.utility.vector1_unsigned_long()
    for index, boolean in enumerate(interface.apply(pose)):
        if boolean and pose.pdb_info().pose2pdb(index + 1).split(" ")[1] == chain1:
            chainAList.append(index + 1)

    for index, boolean in enumerate(interface.apply(pose)):
        if boolean and pose.pdb_info().pose2pdb(index + 1).split(" ")[1] == chain2:
            chainBList.append(index + 1)

    chainAInterfaceResidues = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector(chainAList)
    chainBInterfaceResidues = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector(chainBList)

    return [chainAInterfaceResidues, chainBInterfaceResidues]


def make_chunks(data: list, thread_count) -> dict:
    """
    Takes a list and splits it into parts based on the thread count
    :param data: a list that needs to be split up amongst the threads
    :param thread_count: the number of threads to use
    :return: None
    """
    threads = {}

    for x in range(0, thread_count):
        threads[x] = []

    thread = 0
    for x in range(0, len(data)):
        threads[thread].append(data[x])
        thread += 1
        if thread == thread_count:
            thread = 0
    return threads
