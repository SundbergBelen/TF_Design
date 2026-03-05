#!/usr/bin/env python3
import sys
class StdoutTagger:
    def write(self, msg):
        if msg.strip():
            for line in msg.splitlines(True):
                sys.__stdout__.write(f"[PYTHON STDOUT] {line}")
    def flush(self):
        sys.__stdout__.flush()

class StderrTagger:
	def write(self, msg):
		if msg.strip():
			sys.__stderr__.write(f"[ROSETTA STDERR] {msg}")
	def flush(self):
		sys.__stderr__.flush()


import os
import csv
import glob
import psutil
import argparse
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
import pandas as pd
import tempfile


# ============================================================
# === USER SETTINGS — MODIFY THESE ONLY ======================
# ============================================================

input_folder = "/ifs/share/TF_Project/Design/TF_project_design_step/GalS/Design_input"
directory_name = "/ifs/scratch/home/bs3281/TF_Project_BS/Test"
run_name = "Output_backrub_1k_test_low_best"
residues_to_mutate = [275, 276, 277, 278, 279, 280, 281]
n_struct_backrub=5
n_trials_backrub=1000

# Pipeline behavior flags
USE_ProteinMPNN   = True
USE_RosettaDesign = False
USE_relaxed_input = True # are the inputs already relaxed? 
USE_backrub       = True
BIAS_FLAG         = False
TIED_FLAG         = True

# Design parameters
N_TRIALS = 3
N_PASS   = 2

# Parallelism
NUM_THREADS = 1 #If set to more than 1 then the full log is not printed.


# ============================================================
# === PATHS SETUP ============================================
# ============================================================

protein_mpnn_repo = Path("/ifs/scratch/home/bs3281/Parrots_BS/ProteinMPNN")
protein_mpnn_parent = protein_mpnn_repo.parent

if str(protein_mpnn_parent) not in sys.path:
	sys.path.insert(0, str(protein_mpnn_parent))

from utilities import *
import energy_methods_original
from ProteinMPNN import protein_mpnn_run


# ============================================================
# === MAIN PIPELINE FUNCTION =================================
# ============================================================
def execute_pipeline(
	pdb_paths,
	protein_mpnn_1_path,
	protein_mpnn_2_path,
	backrub_path,
	temp_dir_mpnn,
	temp_dir_rosetta,
	n_thread,
	run_dir,
):
	"""Execute PARROTS pipeline for a given list of PDBs using global user settings."""
	
	# log_file = os.path.join(run_dir, f"log_files/worker_{n_thread}.log")
	# sys.stdout = open(log_file, "w")
	# sys.stderr = sys.stdout

	print(f"Worker {n_thread} starting", flush=True)

	import pyrosetta
	pyrosetta.init(
		extra_options="-relax:default_repeats 1 -ignore_zero_occupancy false -multithreading:total_threads 1 -constant_seed false"
	)
	
	print(f"[PID {os.getpid()}] PyRosetta initialized", flush=True)

	import tempfile
	from pathlib import Path

	# Create unique temp root for this worker
	temp_root = Path(run_dir) / "temp_files"
	temp_root.mkdir(exist_ok=True)

	worker_temp_root = Path(
		tempfile.mkdtemp(prefix=f"worker{n_thread}_", dir=temp_root)
	)

	temp_dir_mpnn = worker_temp_root / "mpnn"
	temp_dir_rosetta = worker_temp_root / "rosetta"

	temp_dir_mpnn.mkdir(exist_ok=True)
	temp_dir_rosetta.mkdir(exist_ok=True)

	print(f"[INFO] Worker {n_thread} temp root: {worker_temp_root}", flush=True)
		
	# Assign CPU affinity
	proc = psutil.Process()
	try:
		proc.cpu_affinity([n_thread, n_thread + 1])
	except Exception as e:
		print(f"[WARNING] Could not set CPU affinity for thread {n_thread}: {e}", flush=True)
	
	if not pdb_paths:
		print(f"[INFO] Thread {n_thread} received no PDBs — skipping.", flush=True)
		return []

	res_to_mutate_list = residues_to_mutate
	chain_res_design_dict = {"A": res_to_mutate_list, "B": res_to_mutate_list}

	pre_step_1 = pdb_paths
	print(f"[INFO.execute_pipeline] pdb_paths: {pdb_paths}", flush=True)

	# Backrub stage
	if USE_backrub:
		print(f"[INFO.execute_pipeline] USE BACKRUB TRUE", flush=True)
		backrub_designs = energy_methods_original.perform_chainA_backrub(
			pdb_paths,
			backrub_path,
			n_struct_backrub=n_struct_backrub,
			n_trials_backrub=n_trials_backrub,
			chain_res_design_dict=chain_res_design_dict,
		)
		pre_step_1 = backrub_designs

	# -------- Step 1 --------
	print(f"\n[INFO.execute_pipeline] Starting Design Round 1", flush=True)
	step_1_passed = energy_methods_original.design_round_for_WT(
		pre_step_1,
		protein_mpnn_1_path,
		USE_ProteinMPNN,
		USE_RosettaDesign,
		USE_relaxed_input,
		n_thread,
		N_TRIALS,
		N_PASS,
		BIAS_FLAG,
		TIED_FLAG,
		chain_res_design_dict,
		temp_dir_mpnn=str(temp_dir_mpnn),
		temp_dir_rosetta=str(temp_dir_rosetta),
	)

	print(f"[INFO.execute_pipeline] Step 1 passed: {step_1_passed}", flush=True)

	step_1_list = []
	for file in step_1_passed:
		entry = {"name": file}
		try:
			entry = energy_methods_original.get_dgDSASA_dict(file, entry)
		except Exception as e:
			print(f"[ERROR.execute_pipeline] Step 1 failed for {file}: {e}", flush=True)
			continue
		step_1_list.append(entry)

	if step_1_list:
		pd.DataFrame(step_1_list).to_csv(csv_step1, mode="a", index=False, header=False)

	# -------- Step 2 --------
	print(f"\n[INFO.execute_pipeline] Starting Design Round 2", flush=True)
	step_2_passed = energy_methods_original.design_round_for_WT(
		step_1_passed,
		protein_mpnn_2_path,
		USE_ProteinMPNN,
		USE_RosettaDesign,
		True,  # step2 always uses relaxed inputs
		n_thread,
		N_TRIALS,
		N_PASS,
		BIAS_FLAG,
		TIED_FLAG,
		chain_res_design_dict,
		temp_dir_mpnn=str(temp_dir_mpnn),
		temp_dir_rosetta=str(temp_dir_rosetta),
	)

	print(f"[INFO.execute_pipeline] Step 2 passed: {step_2_passed}", flush=True)

	step_2_list = []
	for file in step_2_passed:
		entry = {"name": file}
		try:
			entry = energy_methods_original.get_dgDSASA_dict(file, entry)
		except Exception as e:
			print(f"[ERROR.execute_pipeline] Step 2 failed for {file}: {e}", flush=True)
			continue
		step_2_list.append(entry)

	if step_2_list:
		pd.DataFrame(step_2_list).to_csv(csv_step2, mode="a", index=False, header=False)

	return step_1_passed


# ============================================================
# === MAIN ====================================================
# ============================================================
def main():
	# ============================================================
	# PRINT SETTINGS
	# ============================================================
	print("\n========== PIPELINE CONFIGURATION ==========")
	print(f"Input folder:          {input_folder}")
	print(f"Output base directory: {directory_name}")
	print(f"Run name:              {run_name}")

	print("--- Flags ---")
	print(f"ProteinMPNN enabled:   {USE_ProteinMPNN}")
	print(f"Rosetta Design:        {USE_RosettaDesign}") #ie. after mpnn.
	print(f"Relaxed inputs:        {USE_relaxed_input}")
	print(f"Backrub enabled:       {USE_backrub}")
	print(f"Bias flag:             {BIAS_FLAG}")
	print(f"Tied flag:             {TIED_FLAG}")
	
	print("--- Backrub Parameters ---")
	print(f"Backrub enabled:       {USE_backrub}")
	print(f"n struct backrub:      {n_struct_backrub}")
	print(f"n trials backrub:      {n_trials_backrub}")

	print("--- Design Parameters ---")
	print(f"n_trials:              {N_TRIALS}")
	print(f"n_pass:                {N_PASS}")
	print(f"num_threads:           {NUM_THREADS}")
	print("============================================\n")

	pattern = os.path.join(input_folder, "*.pdb")
	list_of_binders = glob.glob(pattern)
	print(f"[INFO] Found {len(list_of_binders)} PDB files")

	num_threads = min(NUM_THREADS, len(list_of_binders))

	chunks = make_chunks(list_of_binders, num_threads)
	print(f"[INFO] Num chunks: {chunks}")
	threads = []

	# Output directories
	base_output = Path(directory_name)
	run_dir = base_output / run_name
	scoring_dir = run_dir / "scoring"
	step1_dir = run_dir / "step1"
	step2_dir = run_dir / "step2"
	backrub_dir = run_dir / "backrub_outputs"
	# log_dir = run_dir / "log_files"

	for d in [base_output, run_dir, scoring_dir,step1_dir, step2_dir, backrub_dir,]:
		make_dir(d)

	protein_base_name = os.path.basename(directory_name)
	csv_suffix = f"_{protein_base_name}_{run_name}.csv"

	global csv_step1, csv_step2
	csv_step1 = str(scoring_dir / f"step_1_design{csv_suffix}")
	csv_step2 = str(scoring_dir / f"step_2_design{csv_suffix}")

	# Initialize CSVs
	for csv_file in [csv_step1, csv_step2]:
		if not os.path.exists(csv_file):
			with open(csv_file, "w", newline="") as f:
				writer = csv.writer(f)
				writer.writerow(energy_methods_original.get_dgDSASA_keys())

	# Spawn worker threads
	for thread_num in range(num_threads):

		args = (
			chunks[thread_num],
			str(step1_dir),
			str(step2_dir),
			str(backrub_dir),
			None,  # temp_dir_mpnn (handled inside execute_pipeline)
			None,  # temp_dir_rosetta (handled inside execute_pipeline)
			thread_num,
			str(run_dir),
		)

		p = mp.Process(target=execute_pipeline, args=args)
		threads.append(p)
		p.start()

	for t in threads:
		t.join()

		print(f"\n✅ All outputs stored in: {run_dir}\n")


if __name__ == "__main__":
	main()
