import os, sys
import subprocess
import pandas
import pandas as pd
from utilities import *
import pyrosetta
import csv
from pyrosetta import init, pose_from_file
from protein_mpnn_mutator_original import protein_mpnn_designs
from pyrosetta.rosetta.core.select.residue_selector import ChainSelector
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.core.pack.task.operation import InitializeFromCommandline, RestrictResidueToRepacking
from pyrosetta.rosetta.protocols.task_operations import RestrictChainToRepackingOperation
from pyrosetta.rosetta.core.pack.task.operation import RestrictToRepacking
from pyrosetta.rosetta.protocols.backrub import BackrubMover
from pyrosetta.rosetta.core.select.movemap import MoveMapFactory
from pyrosetta.rosetta.protocols.simple_filters import ContactMolecularSurfaceFilter
from pyrosetta.rosetta.core.scoring import CA_rmsd
import time


# print("===== ENV DEBUG =====")
# print("Python:", sys.executable)
# print("PyRosetta version:", pyrosetta.__file__)
# print("Working dir:", os.getcwd())
# print("OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"))
# print("SLURM_JOB_ID:", os.environ.get("SLURM_JOB_ID"))
# print("=====================")


def design_round_for_WT(
	pdb_list: list,
	output_path: os.PathLike,
	protein_mpnn_flag: bool = True,
	design_flag: bool = True,
	relaxed_flag: bool = True,
	thread_num: int = 0,
	n_trials: int = 8,
	n_pass: int = 4,
	bias_flag: bool = True,
	tied_flag: bool = False,
	chain_res_design_dict: dict = None,
	temp_dir_mpnn: str = None,
	temp_dir_rosetta: str = None,
) -> list:

	"""

	Generates a given amount of models and saves the top scoring n models using complex_interface/dSASAx100 score as a reference, returns a list of paths of the scoring models.
	@param output_path:
	@param n_pass: int, get n number of top models from models generated
	@param n_trials:  int, models to generate and score
	@param pdb_list: list, pdb paths for starting files
	@param protein_mpnn_flag: bool, protein_mpnn method on or off
	@param design_flag: bool, Rosetta Design on or off
	@param relaxed_flag: bool, are the inputs relax?
	@param thread_num: if multithreading, send thread number
	@param bias_flag: if True uses bias for sampling
	@param tied_flag: if True, ties positions for homooligomer design
	@param chain_res_design_dict: dictionary including residues to design for each chain

	@return: list, file paths for all mutated models
	"""
	from protein_mpnn_mutator_original import protein_mpnn_designs

	log(f"Starting design round...")
	log(f"PDB list: {pdb_list}")
	make_dir(output_path)
	pdb_output_path = os.path.join(output_path, "pdbs")
	make_dir(pdb_output_path)

	if protein_mpnn_flag:
		chain_path: str = os.path.join(output_path, 'chains')
		seq_path: str = os.path.join(output_path, "sequences")
		make_dir(chain_path)
		make_dir(seq_path)
	#first_residue_helical_bundle = 532  # always the same

	count = 0
	complete_solution = []   # <-- always initialize

	# If no PDBs, return empty list early
	if len(pdb_list) == 0:
		log("No PDBs passed into design_round_for_WT()", level = ' WARNING ')
		return complete_solution
	for pdb_path in pdb_list:
		log(f"Starting design for: {pdb_path.split('/')[-1]}")
		# Set which residues are to be mutated:
		# Pseudocode of the following lines of code:
		# Make residue selector from first residues in HSA (Chain A) to residue 532 (Residue where we make the cut to
		# add helical bundles)
		# Make a residue selector for the antigen (Chain B)
		# Create immutable residue selector, which is the combination of the HSA residues and the antigen residuesP
		# Create the mutable position, which is every residue that is not in immutable residues.
		# Notes:
		# All HA inputs do not have an interface between the HSA and the antigen, therefore we are only mutating the
		# helical bundle.
		# For future targets, you just need to add a neighbor residue between chain A and chain B and add it to the list
		# of mutable residues.
		
		hsa_pose: pyrosetta.Pose = pyrosetta.pose_from_file(pdb_path)
		info: pyrosetta.rosetta.core.pose.PDBInfo = hsa_pose.pdb_info()

		'''
		residue_list = [541, 542, 543, 546, 549, 550, 553, 556, 557, 560, 561, 562, 563, 564, 566, 569, 570, 571, 574, 577, 578, 581]
		pose_list_str = ",".join(str(info.pdb2pose("A", res)) for res in residue_list)
		pose_res_list = [info.pdb2pose("A", res) for res in residue_list]
		print("Pose res list", pose_res_list)
		print(pose_list_str)
		'''

		'''
		Design oligomer
		chain_res_design_dict: for each chain, design residues in list
		'''
		#res_to_mutate_list = [277,278,279,280,281,282,283] # original list
		#res_to_mutate_list = [221, 223, 251, 254, 255, 277, 278, 279, 280, 281, 282, 283] # new list with added residues from other helices
		#chain_res_design_dict = {'A': res_to_mutate_list, 'B': res_to_mutate_list}

		pose_list = []
		mutable_list = [] # mutable list for mpnn
		for chain_id, res_list in chain_res_design_dict.items():
			mutable_list.append([])
			for res in res_list:
				pose_list.append(str(info.pdb2pose(chain_id, res)))
				mutable_list[-1].append(str(info.pdb2pose('A', res))) # use the numbering for chain A (mpnn starts at res id 1 for each chain)

		dprint(f'Mutable list: {mutable_list}')

		dprint(f'Pose list: {pose_list}')
		pose_list_str = ",".join(pose_list)
		log(f"pose_list_str: {pose_list_str}")

		#residue_list.sort()
		
		hsa_res_selector= pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector(pose_list_str)
		log(f'Residue selector: {hsa_res_selector.selection_positions(hsa_pose)}')
		#immutable_residues: pyrosetta.rosetta.core.select.residue_selector.NotResidueSelector = pyrosetta.rosetta.core.select.residue_selector.NotResidueSelector()
		#immutable_residues.set_residue_selector(hsa_res_selector)
		#print("immutable: ", immutable_residues.selection_positions(hsa_pose))
		#mutable_residue_list = get_mutable_list_from_residue_selector(hsa_res_selector,hsa_pose, pyrosetta.rosetta.core.pose.get_chain_id_from_chain("A", hsa_pose))

		# Create individual file directories
		pdbs_output = os.path.join(pdb_output_path, pdb_path.split("/")[-1][0:-4])
		make_dir(pdbs_output)
		if protein_mpnn_flag:
			log(f"Protein MPNN Flag: True")
			models_chain_path = os.path.join(chain_path, pdb_path.split("/")[-1][0:-4])
			models_seq_path = os.path.join(seq_path, pdb_path.split("/")[-1][0:-4])
			make_dir(models_chain_path)
			make_dir(models_seq_path)
			# Calls function depending on input flags for design methods.
			log(f"Starting Protein MPNN sequence generation: choose {n_pass} best structures from {n_trials} sequences.")
			passed = protein_mpnn_designs(
					pdb_path,
					models_chain_path,
					models_seq_path,
					pdbs_output,
					" ".join(list(chain_res_design_dict.keys())),
					hsa_res_selector,
					mutable_list,
					relaxed_flag,
					design_flag,
					thread_num,
					n_trials,
					n_pass,
					bias_flag,
					tied_flag,
					temp_dir=temp_dir_mpnn,
				)
		else:
			log(f"Protein MPNN Flag: False")
			passed = get_designs(pdb_path, hsa_res_selector, pdbs_output, n_trials, n_pass,
								 thread_num)
		if len(pdb_list) == 1:
			complete_solution = passed
		else:
			for x in passed:
				complete_solution.append(x)
		count = count + 1
		log(f"energy methods count: {count}")
		log(f"Complete solution: {complete_solution}")
	return complete_solution


def get_designs(pdb_path: os.path, residue_selector: pyrosetta.rosetta.core.select.residue_selector.ResidueSelector,
				design_output_folder: os.path, n_trials: int, n_pass: int, thread_num: int, temp_dir_rosetta: str) -> list:
	"""
	Generates a defined number of Rosetta design model, only residues in residue selector are allowed to design.
	It then scores all models using complex_interface/dSASAx100 and n_pass lower scoring models. Returns a
	list of path to passed models.

	@param pdb_path: str, input path to design
	@param residue_selector: pyrosetta.rosetta.core.select.residue_selector.Residue_selector, residue selector of residues to design
	@param design_output_folder: str, folder where to send the outputs
	@param n_trials: number of models to generate
	@param n_pass: number of models to pass, ranked from lowest to highest scores
	@param thread_num: if multithreading, send thread number
	@return: list of paths to pass models
	"""
	log(f"Generating Rosetta models and filtering by complex_interface/dSASAx100")
	log(f"pdb_path: {pdb_path}")
	designs = []
	file = pdb_path.split("/")[-1][0:-4]
	working_pose = pyrosetta.pose_from_pdb(pdb_path)

	mut_dict = {}
	for i in range(n_trials):
		current_design = relax_with_design(working_pose, residue_selector)
		name_file = os.path.join(design_output_folder, str(file) + "_" + str(i) + "_model.pdb")
		pdb_name_temp = str(i) + str(thread_num) + ".pdb"
		
		import uuid

		unique_id = uuid.uuid4().hex[:6]
		pdb_temp_path = os.path.join(
			temp_dir_rosetta,
			f"{file}_thread{thread_num}_trial{i}_{unique_id}.pdb")
		
		current_design.dump_pdb(pdb_temp_path)
		log(f"Dumped temp pdb: {pdb_temp_path}")
		temp_dict = {}
		new = get_dgDSASA_dict(os.path.join("temp_files_rosetta", pdb_name_temp), temp_dict)[
			'separated_interface/dSASAx100']
		mut_dict[i] = [current_design, name_file, new]
	keys = get_keys_with_lowest_scores(mut_dict, n_pass)
	for i in keys:
		output_path = os.path.abspath(mut_dict[i][1])
		mut_dict[i][0].dump_pdb(output_path)
		log(f"Dumped final pdb: {output_path}")
		designs.append(output_path)
	return designs


def relax_with_design(Pose: pyrosetta.Pose,
					  residue_selectorA: pyrosetta.rosetta.core.select.residue_selector.ResidueSelector
					  , iterations: int = 80, repacking_distance: int = 10,
					  back_bone_flag: bool = False) -> pyrosetta.Pose:
	"""
	Creates a model using the Rosetta design algorithm from a given pose and residue selector. It repacks given selector and all residues within a given
	packing radius, default is 10 A.
	@param Pose: pyrosetta.Pose, pose to design
	@param residue_selectorA: Residues to design
	@param iterations: relax algorithm iterations, default 80
	@param repacking_distance:  distance to repack around given residue selector, default is 10 A.
	@param back_bone_flag:  Restrict relax backbone movements, default is False
	@return: pyrosetta.Pose
	"""
	log(f"Starting relax with design...")
	score_fxn = pyrosetta.get_fa_scorefxn()
	working_pose: pyrosetta.rosetta.core.pose.Pose = Pose.clone()
	score_before = score_fxn(working_pose)
	log(f"Score before relax: {score_before:.3f}")
	design = pyrosetta.rosetta.protocols.relax.FastRelax(standard_repeats=1)
	#design.cartesian(True)
	design.set_scorefxn(score_fxn)
	tf = pyrosetta.rosetta.core.pack.task.TaskFactory()
	tf.push_back(pyrosetta.rosetta.core.pack.task.operation.InitializeFromCommandline())
	restrict_to_repacking = pyrosetta.rosetta.core.pack.task.operation.RestrictToRepackingRLT()

	repacking_area = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector()
	repacking_area.set_focus_selector(residue_selectorA)
	repacking_area.set_distance(10)
	repacking_area.set_include_focus_in_subset(True)
	no_repacking = pyrosetta.rosetta.core.pack.task.operation.PreventRepackingRLT()
	prevent_residue_from_designing = pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(
		restrict_to_repacking,
		residue_selectorA, True)
	tf.push_back(prevent_residue_from_designing)

	prevent_residues_from_repacking = pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(no_repacking,
																										repacking_area,
																										True)
	tf.push_back(prevent_residues_from_repacking)

	design.constrain_relax_to_start_coords(back_bone_flag)
	design.max_iter(iterations)
	movemap = pyrosetta.MoveMap()
	movemap.set_bb(repacking_area.apply(Pose))
	movemap.set_chi(repacking_area.apply(Pose))
	movemap.set_jump(repacking_area.apply(Pose))
	design.set_scorefxn(score_fxn)
	design.set_movemap(movemap)
	design.set_task_factory(tf)
	design.apply(working_pose)

	score_after = score_fxn(working_pose)
	score_delta = score_after - score_before

	log(f"Score after relax:  {score_after:.3f}")
	log(f"Delta score:        {score_delta:.3f}")

	return working_pose

def calculate_contact_surface(pose: pyrosetta.Pose):
	#pyrosetta.init()

	log(f"Calculating contact surface area...")
	
	try:
		# Define chains A and B
		chainA = ChainSelector('A')  
		chainB = ChainSelector('B')  

		chain_A_selector = pyrosetta.rosetta.core.select.get_residue_selector_from_subset(chainA.apply(pose))
		chain_B_selector = pyrosetta.rosetta.core.select.get_residue_selector_from_subset(chainB.apply(pose))
		
		# define the filter ContactMolecularSurfaceFilter
		contact_surface_filter = ContactMolecularSurfaceFilter()
		
		# Set selectors for the filter using selector1 and selector2 methods
		contact_surface_filter.selector1(chain_A_selector)
		contact_surface_filter.selector2(chain_B_selector)
		
		# Set chain B as the apolar target
		contact_surface_filter.apolar_target(True)  # True indicates chain B is apolar target
		
		# Apply the filter to compute contact molecular surface
		contact_surface_value = contact_surface_filter.compute(pose)
	except Exception as e:
		log(f"Error processing file {pose}: {e}")
		
	return contact_surface_value

def get_dgDSASA_keys() -> list:
	"""
	Used to make the header for step_1_design and step_2_design csv files. 
	Called in Design_and_Score when initially making csv files with headers.

	@return: list with all header names (new values/header names can easily be added to this)
	"""
	dgDSASA = []
	if not dgDSASA:
		dgDSASA = ['name',
					'separated_interface',
					'complex_interface_sum',
					'dSASA',
					'complex_interface/dSASA',
					'complex_interface/dSASAx100',
					'separated_interface/dSASA',
					'separated_interface/dSASAx100',
					'cross_hbond',
					'hbond_energy/separated_interface',
					'unsat_hbond',
					'contact_molecular_surface',
					'pairwise_interface_energy',
					'positive_pairwise_interface_energy_sum',
					'pairwise_energies',
					]
	return dgDSASA


def relax(Pose: pyrosetta.Pose,
		  residue_selectorA: pyrosetta.rosetta.core.select.residue_selector = pyrosetta.rosetta.core.select.residue_selector.TrueResidueSelector(),
		  iterations= 80, repacking_distance: int = 10, back_bone_flag: bool = False) -> pyrosetta.Pose:
	"""
	Local relax of given pose and a residue selector. It repacks given selector and all residues within a given
	packing radius, default is 10 A.

	@param Pose: pyrosetta.Pose, pose to relax
	@param residue_selectorA: residue_selector, residue selector to relax and relax around. default is all residues.
	@param iterations: Iterations for relax algorithm, default is 80
	@param repacking_distance:  distance to repack around given residue selector, default is 10 A.
	@param back_bone_flag:  Restrict relax backbone movements, default is False
	@return: pyrosetta.Pose, relaxed pose
	"""
	
	score_fxn = pyrosetta.get_fa_scorefxn()
	working_pose: pyrosetta.rosetta.core.pose.Pose = Pose.clone()

	selected_subset = residue_selectorA.apply(Pose)
	selected_residues = pyrosetta.rosetta.core.select.get_residues_from_subset(selected_subset)

	log(f"Running local relax (residues: {selected_residues}, iterations: {iterations}, repacking_distance: {repacking_distance}, back_bone_flag: {back_bone_flag})")

	score_before = score_fxn(working_pose)


	relax_mover = pyrosetta.rosetta.protocols.relax.FastRelax(standard_repeats=1)
	relax_mover.set_scorefxn(score_fxn)
	tf = pyrosetta.rosetta.core.pack.task.TaskFactory()
	tf.push_back(pyrosetta.rosetta.core.pack.task.operation.InitializeFromCommandline())
	restrict_to_repacking = pyrosetta.rosetta.core.pack.task.operation.RestrictToRepacking()
	repacking_area = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector()
	repacking_area.set_focus_selector(residue_selectorA)
	repacking_area.set_distance(repacking_distance)
	repacking_area.set_include_focus_in_subset(True)
	tf.push_back(restrict_to_repacking)
	no_repacking = pyrosetta.rosetta.core.pack.task.operation.PreventRepackingRLT()
	prevent_residues_from_repacking = pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(no_repacking,
																										repacking_area,
																										True)
	relax_mover.constrain_relax_to_start_coords(back_bone_flag)
	tf.push_back(prevent_residues_from_repacking)
	relax_mover.max_iter(iterations)
	movemap = pyrosetta.MoveMap()
	movemap.set_bb(repacking_area.apply(Pose))
	movemap.set_chi(repacking_area.apply(Pose))
	movemap.set_jump(repacking_area.apply(Pose))
	relax_mover.set_scorefxn(score_fxn)
	relax_mover.set_movemap(movemap)
	relax_mover.set_task_factory(tf)
	relax_mover.apply(working_pose)

	score_after = score_fxn(working_pose)
	score_delta = score_after - score_before
	
	log(f"Score before relax: {score_before}")
	log(f"Score after relax: {score_after:.3f}")
	log(f"score_after - score_before: {score_delta:.3f}")

	return working_pose


def perform_chainA_backrub(pdb_list: list, backrub_output: os.path, n_struct_backrub=10, n_trials_backrub=1000,chain_res_design_dict={}):
	log(f"PERFORMING perform_chainA_backrub IN ENERGY METHODS")
	#init()
	make_dir(backrub_output)
	final: list = []
	index = 0
	log(f"chain_res_design_dict {chain_res_design_dict}")

	for pdb_path in pdb_list:
		file = pdb_path.split("/")[-1][0:-4]
		log(f"DB PATH: {pdb_path}")
		# print(f"[INFO.perform_chainA_backrub] FILE: {file}")

		working_pose: pyrosetta.Pose = pyrosetta.pose_from_file(pdb_path)
		info: pyrosetta.rosetta.core.pose.PDBInfo = working_pose.pdb_info()
		pose_list = []
		for chain_id, res_list in chain_res_design_dict.items():
			for res in res_list:
				pose_list.append(str(info.pdb2pose(chain_id, res)))

		log(f"Pose list: {pose_list}")
		pose_list_str = ",".join(pose_list)
		
		hsa_res_selector= pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector(pose_list_str)

		nbr_selector = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector()
		nbr_selector.set_focus_selector(hsa_res_selector)
		nbr_selector.set_distance(10.0)

		nbr_subset = nbr_selector.apply(working_pose)
		nbr_resnums = pyrosetta.rosetta.core.select.get_residues_from_subset(nbr_subset)

		log("Neighborhood residues:", nbr_resnums)

		mmf = pyrosetta.rosetta.core.select.movemap.MoveMapFactory()
		
		mmf.add_bb_action(pyrosetta.rosetta.core.select.movemap.move_map_action.mm_enable, hsa_res_selector)
		mmf.add_bb_action(pyrosetta.rosetta.core.select.movemap.move_map_action.mm_enable, nbr_selector)

		br_mover = pyrosetta.rosetta.protocols.backrub.BackrubMover()
		br_mover.set_movemap_factory(mmf)

		# PRIME THE MOVER
		log(f"Priming backrub mover (defaults to all residues)...") # backrub needs to be applied before you add segments...
		temp_pose = pyrosetta.Pose(working_pose)
		br_mover.apply(temp_pose)

		log(f"Clearing segments...")
		br_mover.clear_segments()

		from pyrosetta.rosetta.core.id import AtomID

		subset = hsa_res_selector.apply(working_pose)
		resnums = pyrosetta.rosetta.core.select.get_residues_from_subset(subset)
		start_resnum = min(resnums)
		end_resnum = max(resnums)

		log("Selected residues (pose numbering):", resnums)

		for i in resnums:
			if i > 1 and i < working_pose.total_residue():
				ca1 = working_pose.residue(i-1).atom_index("CA")
				ca2 = working_pose.residue(i+1).atom_index("CA")

				br_mover.add_segment(
					AtomID(ca1, i-1),
					AtomID(ca2, i+1)
				)

		log(f"Num segments from neighborhood selector: {br_mover.num_segments()}")
		# br_protocol = pyrosetta.rosetta.protocols.backrub.BackrubProtocol()
		# print(f"[INFO.perform_chainA_backrub] backrub mover type: {type(br_mover)}")

		scorefxn = pyrosetta.get_fa_scorefxn()
		score_input = scorefxn(working_pose)

		log(f"Num backrub structures to output: {n_struct_backrub}")
		lowest_score = float("inf")
		lowest_pose = None
		last_pose = None

		score_rows = []

		input_score = scorefxn(working_pose)

		score_rows.append({
			"structure_type": "input",
			"trajectory": -1,
			"pdb_path": str(file),
			"score": input_score})

		for i in range(n_struct_backrub):

			pose_copy = pyrosetta.Pose(working_pose)

			# --- Prove raw backrub actually moves atoms before MC ---
			pose0 = pyrosetta.Pose(pose_copy)

			for _ in range(5):
				br_mover.apply(pose_copy)

			first_rmsd = CA_rmsd(pose0, pose_copy, start_resnum, end_resnum) #rn just for one helix. 

			if first_rmsd < 5e-3:
				raise RuntimeError(
					f"[INFO.perform_chainA_backrub] Backrub proposals are effectively zero "
					f"(ΔRMSD after 5 raw moves = {first_rmsd:.4f} Å). "
					"BB DOFs likely blocked or pivots invalid."
				)

			# reset pose before MC
			pose_copy.assign(pose0)

			mc = pyrosetta.rosetta.protocols.moves.MonteCarlo(
				pose_copy,
				scorefxn,
				1.5
			)

			accepted = 0
			start = time.time()

			log(f"Running {n_trials_backrub} MC trials...")

			print_interval = max(1, n_trials_backrub // 20)

			# --- reset per-trajectory tracking ---
			lowest_score = scorefxn(pose_copy)
			lowest_pose = pose_copy.clone()

			for step in range(n_trials_backrub):

				br_mover.apply(pose_copy)

				if mc.boltzmann(pose_copy):
					accepted += 1

				current_score = scorefxn(pose_copy)

				if current_score < lowest_score:
					lowest_score = current_score
					lowest_pose = pose_copy.clone()

				if step % print_interval == 0 or step == n_trials_backrub - 1:
					log(
						f"MC step {step}/{n_trials_backrub} "
						f"({100*step/n_trials_backrub:.1f}%)  "
						f"Accepted: {accepted}",
					)

			end = time.time()

			final_rmsd = CA_rmsd(pose0, pose_copy, start_resnum, end_resnum)
			acc_rate = accepted / n_trials_backrub

			log(f"Elapsed time: {end - start:.3f} s")

			log(
				f"accepted={accepted}/{n_trials_backrub} "
				f"({acc_rate*100:4.1f}%)  "
				f"ΔRMSD CA={final_rmsd:.3f} Å  "
				f"MC loop time: {end - start:.2f}s",)

			last_pose = pose_copy.clone()
			last_score = scorefxn(last_pose)

			log(f"delta score (last_score - score_input) (backrub structure {i}) = {last_score - score_input}")

			# --------------------------------------------------
			# Output paths
			# --------------------------------------------------

			last_path = os.path.join(
				backrub_output,
				f"{file}_{index}_br.pdb")

			lowest_path = os.path.join(
				backrub_output,
				f"{file}_{index}_lowest_br.pdb")

			last_pose.dump_pdb(last_path) #only dumping last.
			# lowest_pose.dump_pdb(lowest_path)
			final.append(last_path)

			log(f"Dump last pdb: {last_path}")
			log(f"Dump lowest pdb: {lowest_path}")

			lowest_rmsd = CA_rmsd(pose0, lowest_pose)
			last_rmsd = CA_rmsd(pose0, last_pose)

			log(
				f"lowest energy={lowest_score:.3f} RMSD={lowest_rmsd:.3f} Å | "
				f"last energy={last_score:.3f} RMSD={last_rmsd:.3f} Å"
			)

			score_rows.append({
				"structure_type": "lowest",
				"trajectory": i,
				"pdb_path": lowest_path,
				"score": lowest_score,
				"rmsd": lowest_rmsd
			})

			score_rows.append({
				"structure_type": "last",
				"trajectory": i,
				"pdb_path": last_path,
				"score": last_score,
				"rmsd": last_rmsd
			})

			index += 1

		csv_path = os.path.join(backrub_output, f"{file}_backrub_scores.csv")
		write_backrub_score_csv(csv_path, score_rows)

	return final



def write_backrub_score_csv(csv_path, rows):
    """
    Write score summary for backrub structures.
    
    rows = list of dicts with keys:
        structure_type
        trajectory
        pdb_path
        score
    """
    fieldnames = ["structure_type", "trajectory", "pdb_path", "score", "rmsd"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log(f"Wrote score summary CSV: {csv_path}")


def mut_sequence(new_sequence: str, pose: pyrosetta.rosetta.core.pose.Pose, chain_number: int
				 , Residue_selector: pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector = None
				 , design_flag: bool = False) -> pyrosetta.Pose:
	"""
	Mutates desired sequence into pose, it must be the same size as the len of given chain. It can relax or relax and design,
	It also takes a residue selector to relax around.
	@param new_sequence: str, it contains the new sequence to be mutated.
	@param pose: Input pose
	@param chain_number: rosetta chain ID for chain
	@param Residue_selector: residue selector of residues to relax around.
	@param design_flag: is this relaxing only or also designing? if true it will design
	@return: Mutated pose
	"""
	log(f"Mutating sequence into pose...")
	working_pose: pyrosetta.rosetta.core.pose.Pose = pose.clone()
	'''native_seq_desired_chain = pose.chain_sequence(chain_number)
	first_residue = working_pose.chain_begin(chain_number)
	last_residue = working_pose.chain_end(chain_number)'''
	# for multiple chains
	native_seq = pose.sequence()
	
	#for i, index in enumerate(range(first_residue, last_residue + 1)):
	for i, index in enumerate(range(1, len(native_seq)+1)):
		if native_seq[i] == new_sequence[i]:
			continue
		pyrosetta.toolbox.mutate_residue(working_pose, index, new_sequence[i])

	if design_flag:
		working_pose = relax_with_design(working_pose, Residue_selector)
	else:
		working_pose = relax(working_pose, Residue_selector)
	return working_pose

def get_interface_residues(all_data):
	log(f"Finding interface residues")
	#get interface residues
	interface_residues_vector = all_data.interface_residues
	face_A_vector = interface_residues_vector[2]
	dprint(f'face_A_vector: {face_A_vector}')
	face_B_vector = interface_residues_vector[3]
	dprint(f'face_B_vector: {face_B_vector}')
	#make lists: 
	face_A_list = [i for i, val in enumerate(face_A_vector, start=1) if val]
	dprint(f'face1: {face_A_list}, len: {len(face_A_list)}')
	face_B_list = [i for i, val in enumerate(face_B_vector, start=1) if val]
	dprint(f'face2: {face_B_list}, len: {len(face_B_list)}')

	return face_A_list, face_B_list


def compute_pairwise_interface_energy(pose, face1: list, face2: list, output_csv_path: str = None) -> dict:
	"""
	Computes pairwise interface energies between two sets of residues (face1 and face2).
	If output_csv_path is provided, writes the pairwise energies to a CSV file.
	"""

	log(f"Computing compute_pairwise_interface_energy")
	
	score_fxn = pyrosetta.get_fa_scorefxn()
	score_fxn(pose)

	weights = pose.energies().weights()
	energy_graph = pose.energies().energy_graph()

	vectors = []
	total_energy = 0.0

	for i in face1:
		for j in face2:
			edge = energy_graph.find_energy_edge(i, j)
			if not edge:
				continue

			emap = edge.fill_energy_map()
			pair_energy = emap.dot(weights)

			if pair_energy == 0.0:
				continue  # Skip zero-energy contacts

			resiA = pose.pdb_info().number(i)
			resiB = pose.pdb_info().number(j)

			vectors.append([resiA, resiB, pair_energy])
			total_energy += pair_energy

	# Sort by energy (most negative first)
	vectors.sort(key=lambda x: x[2])

	# Sum of positive energies (destabilizing)
	positive_energy_sum = sum(x[2] for x in vectors if x[2] > 0)

	# Optional CSV output
	if output_csv_path:
		with open(output_csv_path, "w", newline="") as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow(["Residue Chain A", "Residue Chain B", "Pairwise Energy"])
			for resiA, resiB, energy in vectors:
				writer.writerow([resiA, resiB, energy])

	return {
		'vectors': vectors,
		'total_energy': total_energy,
		'positive_energy_sum': positive_energy_sum
	}

def get_dgDSASA_dict(binder: str, current_dict=None)-> dict:
	"""
	Uses the InterfaceAnalyzerMover from Rosetta to calculate the separated_interface, complex_interface_sum, dSASA, and
	combined terms. It uses a pdb file  and it returns a dictionary with score terms.

	@param binder: path to pdb_file
	@param current_dict: dict, if you already have dictionary, it adds scoring terms, and if not, returns a dictionary
			with only the scores
	@return: dictionary with added score terms.
	"""
	working_pose = pyrosetta.pose_from_pdb(binder)
	if current_dict is None:
		current_dict = {}
	try:
		contact_mol_surf = calculate_contact_surface(working_pose)

		interface_analyzer = pyrosetta.rosetta.protocols.analysis.InterfaceAnalyzerMover()
		interface_analyzer.set_pack_separated(True) #this line is necessary to properly calculate separated_interface
		interface_analyzer.apply(working_pose)
		all_data = interface_analyzer.get_all_data()

		face_A_list, face_B_list = get_interface_residues(all_data)

		pairwise_interface_energy, total_energy, positive_energy_sum = compute_pairwise_interface_energy(working_pose, face_A_list, face_B_list).values()
		log(f'Pose: {binder}, Pairwise total energy: {total_energy}, Positive energy sum: {positive_energy_sum}')

				# --- NEW: Worst pairwise energies ---
		# Extract energies only
		energies = [e[2] for e in pairwise_interface_energy]

		# Sort by most positive (worst first)
		worst_sorted = sorted(energies, reverse=True)

		worst_k = 5
		# Take top K worst energies
		worst_k_vals = worst_sorted[:worst_k]

	

		current_dict['separated_interface'] = (all_data.dG)[1] #getting first term in vector
		current_dict['complex_interface_sum'] = all_data.crossterm_interface_energy
		current_dict['dSASA'] = (all_data.dSASA)[1] #getting first term in vector

		current_dict['complex_interface/dSASA'] = all_data.crossterm_interface_energy_dSASA_ratio
		current_dict['complex_interface/dSASAx100'] = (all_data.crossterm_interface_energy_dSASA_ratio) * 100
		current_dict['separated_interface/dSASA'] = all_data.dG_dSASA_ratio
		current_dict['separated_interface/dSASAx100'] = (all_data.dG_dSASA_ratio) * 100
		current_dict['cross_hbond'] = all_data.interface_hbonds
		current_dict['hbond_energy/separated_interface'] = all_data.hbond_E_fraction
		current_dict['unsat_hbond'] = all_data.delta_unsat_hbonds
		current_dict['contact_molecular_surface'] = contact_mol_surf

		# Add pairwise energy data
		current_dict['pairwise_interface_energy'] = total_energy
		current_dict['positive_interface_energy_sum'] = positive_energy_sum
		current_dict['pairwise_energies'] = pairwise_interface_energy  # list of [resiA, resiB, energy]
		current_dict['unfavorable_pairwise_energy_sum'] = sum(worst_k_vals)

	except:
		current_dict['complex_interface/dSASA'] = 0
		current_dict['complex_interface/dSASAx100'] = 0
		current_dict['separated_interface/dSASA'] = 0
		current_dict['separated_interface/dSASAx100'] = 0
		current_dict['cross_hbond'] = 0
		current_dict['hbond_energy/separated_interface'] = 0
		current_dict['unsat_hbond'] = 0
		current_dict['contact_molecular_surface'] = 0
		current_dict['pairwise_interface_energy'] = 0
		current_dict['positive_interface_energy_sum'] = 0
		current_dict['pairwise_energies'] = []
	
	return current_dict
	