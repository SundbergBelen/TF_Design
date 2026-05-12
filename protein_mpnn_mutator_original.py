from utilities import *
import energy_methods_original
import pyrosetta
import os,sys
from argparse import Namespace
from ProteinMPNN import protein_mpnn_run
from ProteinMPNN.helper_scripts import parse_multiple_chains, \
    assign_fixed_chains, make_fixed_positions_dict, make_tied_positions_dict


def safe_get_args(get_args_func):
    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        return get_args_func()
    finally:
        sys.argv = original_argv

def protein_mpnn_maker(
	pdb_path,
	chain_path,
	seq_path,
	chains_to_design='',
	mutable_positions='',
	number_of_predictions=8,
	sorting_temperature=".5",
	bias_jsonl=None,
	tied_flag: bool = False,

) -> str:
    """
    Function calls helper functions from proteinmpnn_glab_version code to generated required input files and stores them in respective folder.
    It calls protein mpnn with given specifications and returns a list with .fa files containing the output sequences.

    Warnings:
    - I don't recommend using this function by itself, it is easier to call protein_mpnn_designs as that function process all the given inputs
    that are compatible with this function.

    - If using bias, change the flag accordingly and modified the dictionary located at proteinmpnn_glab_version/ProteinMPNN_glab/bias_AA.jsonl


    @param pdb_path:str, path to pdb to be proteinmpnn
    @param chain_path: str, path to save input files
    @param seq_path: str, path to save output files
    @param chains_to_design: str, chains to design proteinmpnn flag
    @param mutable_positions: str, mutable positions chains to design proteinmpnn flag
    @param number_of_predictions: int, number of models to be design
    @param sorting_temperature: float , sorting temperature proteinmpnn flag
    @param bias_jsonl: str, path to bias jsonl for mpnn
    @param tied_flag: bool, if True, ties positions for homooligomers
    @return: str, path to .fa files containing the proteinmpnn output sequence.
    """

    log(f"Starting protein mpnn maker...")

    parsed_chain = str(os.path.join(chain_path, pdb_path.split("/")[-1]))[0:-4] + "_parsed_chain.jsonl"
    design_chains = str(os.path.join(chain_path, pdb_path.split("/")[-1]))[0:-4] + "_designed_chain.jsonl"
    kept_residues = str(os.path.join(chain_path, pdb_path.split("/")[-1]))[0:-4] + "_unmutable_chain.jsonl"
    tied_residues = str(os.path.join(chain_path, pdb_path.split("/")[-1]))[0:-4] + "_tied_positions.jsonl"

    args_parsed_chains = safe_get_args(parse_multiple_chains.get_args)
    setattr(args_parsed_chains, "input_path", pdb_path)
    setattr(args_parsed_chains, "output_path", parsed_chain)
    parse_multiple_chains.main(args_parsed_chains)

    if chains_to_design:
        args_designed_chain = safe_get_args(assign_fixed_chains.get_args)
        setattr(args_designed_chain, "input_path", parsed_chain)
        setattr(args_designed_chain, "output_path", design_chains)
        setattr(args_designed_chain, "chain_list", chains_to_design)
        assign_fixed_chains.main(args_designed_chain)

    if mutable_positions:
        args_kept_residues = safe_get_args(make_fixed_positions_dict.get_args)
        setattr(args_kept_residues, "input_path", parsed_chain)
        setattr(args_kept_residues, "output_path", kept_residues)
        setattr(args_kept_residues, "chain_list", chains_to_design)
        setattr(args_kept_residues, "position_list", mutable_positions)
        setattr(args_kept_residues, "specify_non_fixed", True)
        make_fixed_positions_dict.main(args_kept_residues)

    ## tied positions for oligomers
 ## tied positions for oligomers
    if tied_flag:
        # Some ProteinMPNN versions don't expose get_args(); they only have main(parser.parse_args())
        args_tied_positions = None

        # Try common entrypoints in order
        if hasattr(make_tied_positions_dict, "get_args"):
            args_tied_positions = make_tied_positions_dict.get_args()
        elif hasattr(make_tied_positions_dict, "get_parser"):
            parser = make_tied_positions_dict.get_parser()
            args_tied_positions = parser.parse_args([])
        else:
            # Fall back to constructing an argparse.Namespace expected by .main()
            args_tied_positions = Namespace()

        # Set/override fields the script expects
        setattr(args_tied_positions, "input_path", parsed_chain)
        setattr(args_tied_positions, "output_path", tied_residues)
        setattr(args_tied_positions, "homooligomer", 1)

        # Some variants also look for these; harmless if ignored
        if not hasattr(args_tied_positions, "chain_list"):
            setattr(args_tied_positions, "chain_list", "")
        if not hasattr(args_tied_positions, "position_list"):
            setattr(args_tied_positions, "position_list", "")

        make_tied_positions_dict.main(args_tied_positions)

    args_run_proteinmpnn = safe_get_args(protein_mpnn_run.get_args)
    setattr(args_run_proteinmpnn, "pdb_path", pdb_path)
    setattr(args_run_proteinmpnn, "jsonl_path", parsed_chain)
    if chains_to_design:
        setattr(args_run_proteinmpnn, "chain_id_jsonl", design_chains)
    if mutable_positions:
        setattr(args_run_proteinmpnn, "fixed_positions_jsonl", kept_residues)
    if tied_flag:
        setattr(args_run_proteinmpnn, "tied_positions_jsonl", tied_residues)
    setattr(args_run_proteinmpnn, "sampling_temp", sorting_temperature)
    setattr(args_run_proteinmpnn, "num_seq_per_target", number_of_predictions)
    setattr(args_run_proteinmpnn, "out_folder", seq_path)

    if bias_jsonl:
        if not os.path.exists(bias_jsonl):
            raise FileNotFoundError(f"bias_jsonl does not exist: {bias_jsonl}")

        log(f"running mpnn with AA bias: {bias_jsonl}")
        setattr(args_run_proteinmpnn, "bias_AA_jsonl", str(bias_jsonl))
    else:
        log("not using AA bias for mpnn")

    protein_mpnn_run.main(args_run_proteinmpnn)
    return str(pdb_path.split("/")[-1])[0:-4] + ".fa"

def protein_mpnn_designs(
	path_to_pdb,
	chain_path,
	seq_path,
	pdb_output_path,
	chains_to_design='',
	residue_selector=None,
	mutable_list=None,
	relaxed=False,
	design_flag=False,
	thread_num=None,
	n_predictions=8,
	n_pass=4,
	bias_jsonl=None,
	tied_flag=False,
	temp_dir=None,
) -> list:

    """
    Runs protein mpnn for one pdb file, it requires various directories to save required inputs and outputs.
    Important info when using this:
        - ProteinMPNN has its own numbering system, where each chain starts at 1, use function utilities.get_mutable_list_from_residue_selector to get from residue selector to proteinmpnn list
    @param path_to_pdb: str, path to pdb
    @param chain_path: str, path to directory to store required proteinmpnn inputs
    @param seq_path: str, path to directory to store proteinmpnn outputs
    @param pdb_output_path: str, path to directory to story mutated pdb output
    @param chains_to_design: str, chains to design
    @param residue_selector: residue_selector, rosetta residue selector to relax or design around
    @param mutable_list: list of indexes that are mutable (if two or more chains, list of indexes is split into separate list for each chain)
    @param relaxed: are the inputs relax, highly recommend inputs to be relaxed already.
    @param design_flag: should we design with rosetta
    @param thread_num: if multithreading, please send the thread num
    @param n_predictions: how many proteinmpnn predictions should we make and evaulate
    @param n_pass: how many should we pass
    @param bias_jonsl: str to bias jsonl file
    @param tied_flag: if True, ties positions for homooligomer design
    @return: list of paths to passed models.
    """

    file = path_to_pdb.split("/")[-1][0:-4]
    log(f"Starting Protein MPNN design for file: {file}")

    mutable_positions = ", ".join([" ".join(m) for m in mutable_list])

    pose: pyrosetta.rosetta.core.pose.Pose = pyrosetta.pose_from_pdb(path_to_pdb)
    score_fxn: pyrosetta.ScoreFunction = pyrosetta.get_fa_scorefxn()
    
    seq_file = protein_mpnn_maker(
        path_to_pdb,
        chain_path,
        seq_path,
        chains_to_design,
        mutable_positions,
        n_predictions,
        sorting_temperature='0.5',
        bias_jsonl=bias_jsonl,
        tied_flag=tied_flag,
    )

    seq_file_path = os.path.join(seq_path, "seqs", seq_file)
    sequences = get_sequences_from_mpnn(seq_file_path)

    emo = pyrosetta.rosetta.core.scoring.methods.EnergyMethodOptions()
    emo.hbond_options().decompose_bb_hb_into_pair_energies(True)
    score_fxn.set_energy_method_options(emo)
    score_fxn.score(pose)

    if not relaxed:
        log(f"Inputs for mpnn design are not relaxed yet...")
        working_pose: pyrosetta.Pose = energy_methods_original.relax(pose, residue_selector)
    else:
        log(f"Inputs for mpnn design are already relaxed...")
        working_pose: pyrosetta.Pose = pose.clone()

    current_dict = {'file': path_to_pdb.split('/')[-1][0:-4]}
    solutions: list = []
    mut_dict = {}
    for index, sequence in enumerate(sequences):
        name_file = os.path.join(pdb_output_path, str(file) + "_" + str(
            index) + "_model.pdb")
        chain = pyrosetta.rosetta.core.pose.get_chain_id_from_chain("A", pose)

        '''if "/" in sequence:
            sequence = sequence.split("/")[0]'''
        
        # need both sequences of each oligomer to be mutated
        if "/" in sequence:
            sequence = "".join(sequence.split("/")) 

        solution = energy_methods_original.mut_sequence(sequence, working_pose, chain, residue_selector,
                                               design_flag)
        
        if temp_dir is None:
            raise ValueError("WARNING temp_dir must be passed into protein_mpnn_designs()")

        os.makedirs(temp_dir, exist_ok=True)

        # Construct a unique filename for each sequence/model
        base = os.path.splitext(os.path.basename(path_to_pdb))[0]
        
        import uuid
        unique_id = uuid.uuid4().hex[:8]
        
        import uuid

        unique_id = uuid.uuid4().hex[:6]
        pdb_abs_path = os.path.join(
            temp_dir,
            f"{base}_thread{thread_num}_idx{index}_{unique_id}.pdb"
        )

        solution.dump_pdb(pdb_abs_path)

        #old code
        #solution.dump_pdb(os.path.join("temp_files", pdb_name_temp))
        #new = energy_methods_original.get_dgDSASA_dict(os.path.join("temp_files", pdb_name_temp), current_dict)[
            #'separated_interface/dSASAx100']

        #NEW CODE
        log(f"Filtering based on separated_interface/dSASAx100")
        new = energy_methods_original.get_dgDSASA_dict(pdb_abs_path, current_dict)[
        'separated_interface/dSASAx100']
        
        mut_dict[index] = [solution, name_file, new]
        log(f"This is mut_dict[index]: {mut_dict[index]}")

    keys = get_keys_with_lowest_scores(mut_dict, n_pass)

    for i in keys:
        mut_dict[i][0].dump_pdb(mut_dict[i][1])
        solutions.append(mut_dict[i][1])
    log(f"mut_dict[i][1]: {mut_dict[i][1]}",thread=thread_num)
    log(f"solutions: {solutions}", thread=thread_num)
    return solutions
