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
import yaml

def load_config(config_path):
	with open(config_path, "r") as f:
		config = yaml.safe_load(f)

	inputs = config.get("inputs", {})
	options = config.get("options", {})
	environment = config.get("environment", {})

	cfg = {
		"input_folder": inputs["input_folder"],
		"directory_name": inputs["directory_name"],
		"run_name": inputs["run_name"],
		"residues_to_mutate": inputs["residues_to_mutate"],
		"n_struct_backrub": options.get("n_struct_backrub", 2),
		"n_trials_backrub": options.get("n_trials_backrub", 1000),
		"mc_kt": options.get("mc_kt", 0.6),
		"use_protein_mpnn": options.get("use_protein_mpnn", True),
		"use_rosetta_design": options.get("use_rosetta_design", False),
		"use_relaxed_input": options.get("use_relaxed_input", True), #are the inputs already relaxed?
		"use_backrub": options.get("use_backrub", True),
		"bias_jsonl": options.get("bias_jsonl", None),
		"tied_flag": options.get("tied_flag", True),
		"n_trials": options.get("n_trials", 3),
		"n_pass": options.get("n_pass", 2),
		"num_threads": options.get("num_threads", 1),
		"protein_mpnn_repo": Path(environment["protein_mpnn_repo"]),
		
	}

	return cfg

def setup_imports(cfg):
	pipeline_dir = Path(__file__).resolve().parent
	protein_mpnn_repo = cfg["protein_mpnn_repo"]
	protein_mpnn_parent = protein_mpnn_repo.parent

	if str(pipeline_dir) not in sys.path:
		sys.path.insert(0, str(pipeline_dir))

	if str(protein_mpnn_parent) not in sys.path:
		sys.path.insert(0, str(protein_mpnn_parent))

	return protein_mpnn_parent

def execute_pipeline(
	pdb_paths,
	protein_mpnn_1_path,
	protein_mpnn_2_path,
	backrub_path,
	n_thread,
	run_dir,
	cfg,
	csv_step1,
	csv_step2,
):
	from utilities import log
	import energy_methods_original
	import pyrosetta

	log(f"Worker {n_thread} starting")

	pyrosetta.init(
		extra_options=(
			"-relax:default_repeats 1 "
			"-ignore_zero_occupancy false "
			"-multithreading:total_threads 1 "
			"-constant_seed false "
			f"-backrub:ntrials {cfg['n_trials_backrub']} "
			f"-backrub:mc_kt {cfg.get('mc_kt', 0.6)}"
		)
	)

	log(f"[PID {os.getpid()}] PyRosetta initialized")

	temp_root = Path(run_dir) / "temp_files"
	temp_root.mkdir(exist_ok=True)

	worker_temp_root = Path(tempfile.mkdtemp(prefix=f"worker{n_thread}_", dir=temp_root))
	temp_dir_mpnn = worker_temp_root / "mpnn"
	temp_dir_rosetta = worker_temp_root / "rosetta"
	temp_dir_mpnn.mkdir(exist_ok=True)
	temp_dir_rosetta.mkdir(exist_ok=True)

	csv_step1_worker = str(Path(csv_step1).with_name(
	Path(csv_step1).stem + f"_thread{n_thread}" + Path(csv_step1).suffix
	))

	csv_step2_worker = str(Path(csv_step2).with_name(
		Path(csv_step2).stem + f"_thread{n_thread}" + Path(csv_step2).suffix
	))

	log(f"Worker {n_thread} temp root: {worker_temp_root}")

	proc = psutil.Process()
	try:
		proc.cpu_affinity([n_thread, n_thread + 1])
	except Exception as e:
		log(f"Could not set CPU affinity for thread {n_thread}: {e}", level=" WARNING ")

	if not pdb_paths:
		log(f"Thread {n_thread} received no PDBs — skipping.", level=" WARNING ")
		return []

	res_to_mutate_list = cfg["residues_to_mutate"]
	chain_res_design_dict = {"A": res_to_mutate_list, "B": res_to_mutate_list}

	pre_step_1 = pdb_paths
	log(f"pdb_paths: {pdb_paths}")

	if cfg["use_backrub"]:
		if not cfg["use_relaxed_input"]:
			log("Input not relaxed and backrub enabled -> relaxing structure before backrub")
			pre_backrub_paths = []
			for pdb in pdb_paths:
				pose = pyrosetta.pose_from_pdb(pdb)
				relaxed_pose = energy_methods_original.pre_relax_input(pose)
				relaxed_path = os.path.join(backrub_path, f"{Path(pdb).stem}_pre_backrub_relaxed.pdb")
				relaxed_pose.dump_pdb(relaxed_path)
				pre_backrub_paths.append(relaxed_path)
			backrub_input_paths = pre_backrub_paths
		else:
			log("Input already relaxed and backrub enabled -> using input directly for backrub")
			backrub_input_paths = pdb_paths

		# backrub_designs = energy_methods_original.perform_chainA_backrub(
		# 	backrub_input_paths,
		# 	backrub_path,
		# 	n_struct_backrub=cfg["n_struct_backrub"],
		# 	n_trials_backrub=cfg["n_trials_backrub"],
		# 	chain_res_design_dict=chain_res_design_dict,
		# 	write_trajectory=cfg["write_trajectory"],
		# 	trajectory_stride=cfg["trajectory_stride"],
		# 	trajectory_gz=cfg["trajectory_gz"],
		# )
		backrub_designs = energy_methods_original.perform_chainA_backrub_protocol(
			backrub_input_paths,
			backrub_path,
			n_struct_backrub=cfg["n_struct_backrub"],
			chain_res_design_dict=chain_res_design_dict,
		)

		pre_step_1 = backrub_designs
		mpnn_inputs_are_relaxed = True
	else:
		pre_step_1 = pdb_paths
		mpnn_inputs_are_relaxed = cfg["use_relaxed_input"]

	print("\n")
	log("Starting Design Round 1")
	step_1_passed = energy_methods_original.design_round_for_WT(
		pre_step_1,
		protein_mpnn_1_path,
		cfg["use_protein_mpnn"],
		cfg["use_rosetta_design"],
		mpnn_inputs_are_relaxed,
		n_thread,
		cfg["n_trials"],
		cfg["n_pass"],
		cfg["bias_jsonl"],
		cfg["tied_flag"],
		chain_res_design_dict,
		temp_dir_mpnn=str(temp_dir_mpnn),
		temp_dir_rosetta=str(temp_dir_rosetta),
	)

	log(f"Step 1 passed: {step_1_passed}")

	step_1_list = []
	for file in step_1_passed:
		entry = {
			"name": Path(file).stem,
			"path": file,
		}
		try:
			entry = energy_methods_original.get_dgDSASA_dict(
				file,
				entry,
				chain_res_design_dict=chain_res_design_dict,
			)
		except Exception as e:
			log(f"Step 1 failed for {file}: {e}")
			continue
		step_1_list.append(entry)

	if step_1_list:
		step_1_df = pd.DataFrame(step_1_list)

		cols = energy_methods_original.get_dgDSASA_keys()
		step_1_df = step_1_df.reindex(columns=cols)

		step_1_df.to_csv(csv_step1_worker, mode="w", index=False, header=True)

	print("\n")
	log("Starting Design Round 2")
	step_2_passed = energy_methods_original.design_round_for_WT(
		step_1_passed,
		protein_mpnn_2_path,
		cfg["use_protein_mpnn"],
		cfg["use_rosetta_design"],
		mpnn_inputs_are_relaxed,
		n_thread,
		cfg["n_trials"],
		cfg["n_pass"],
		cfg["bias_jsonl"],
		cfg["tied_flag"],
		chain_res_design_dict,
		temp_dir_mpnn=str(temp_dir_mpnn),
		temp_dir_rosetta=str(temp_dir_rosetta),
	)

	log(f"Step 2 passed: {step_2_passed}")

	step_2_list = []
	for file in step_2_passed:
		entry = {
			"name": Path(file).stem,
			"path": file,
		}
		try:
			entry = energy_methods_original.get_dgDSASA_dict(
				file,
				entry,
				chain_res_design_dict=chain_res_design_dict,
			)
		except Exception as e:
			log(f"Step 2 failed for {file}: {e}")
			continue

		step_2_list.append(entry)

	if step_2_list:
		step_2_df = pd.DataFrame(step_2_list)

		cols = energy_methods_original.get_dgDSASA_keys()
		step_2_df = step_2_df.reindex(columns=cols)

		step_2_df.to_csv(csv_step2_worker, mode="w", index=False, header=True)

	return step_1_passed,step_2_passed

def merge_worker_csvs(final_csv, num_threads, dedupe_cols=("name",)):
	final_csv = Path(final_csv)
	worker_csvs = [
		final_csv.with_name(final_csv.stem + f"_thread{i}" + final_csv.suffix)
		for i in range(num_threads)
	]

	dfs = []
	for csv_path in worker_csvs:
		if csv_path.exists() and csv_path.stat().st_size > 0:
			dfs.append(pd.read_csv(csv_path))

	if not dfs:
		return

	merged = pd.concat(dfs, ignore_index=True)

	if dedupe_cols is not None:
		before = len(merged)
		merged = merged.drop_duplicates(subset=list(dedupe_cols), keep="first").copy()
		after = len(merged)
		print(f"[INFO] {final_csv.name}: dropped {before - after} duplicate rows")

	merged.to_csv(final_csv, index=False)

	for csv_path in worker_csvs:
		if csv_path.exists():
			csv_path.unlink()

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", required=True, help="Path to YAML config file")
	args = parser.parse_args()

	cfg = load_config(args.config)
	protein_mpnn_parent = setup_imports(cfg)
	
	from utilities import make_chunks, make_dir, log
	import energy_methods_original
	from ProteinMPNN import protein_mpnn_run  # noqa: F401

	print("\n========== PIPELINE CONFIGURATION ==========")
	print(f"Input folder:          {cfg['input_folder']}")
	print(f"Output base directory: {cfg['directory_name']}")
	print(f"Run name:              {cfg['run_name']}")
	print("--- Flags ---")
	print(f"ProteinMPNN enabled:   {cfg['use_protein_mpnn']}")
	print(f"Rosetta Design:        {cfg['use_rosetta_design']}")
	print(f"Relaxed inputs:        {cfg['use_relaxed_input']}")
	print(f"Backrub enabled:       {cfg['use_backrub']}")
	print(f"Bias jsonl:             {cfg['bias_jsonl']}")
	print(f"Tied flag:             {cfg['tied_flag']}")
	print("--- Backrub Parameters ---")
	print(f"n struct backrub:      {cfg['n_struct_backrub']}")
	print(f"n trials backrub:      {cfg['n_trials_backrub']}")
	print(f"mc_kt backrub:         {cfg.get('mc_kt', 0.6)}")
	print("--- Design Parameters ---")
	print(f"n_trials:              {cfg['n_trials']}")
	print(f"n_pass:                {cfg['n_pass']}")
	print(f"num_threads:           {cfg['num_threads']}")
	print("============================================\n")

	pattern = os.path.join(cfg["input_folder"], "*.pdb")
	list_of_binders = glob.glob(pattern)
	log(f"Found {len(list_of_binders)} PDB files")

	num_threads = min(cfg["num_threads"], len(list_of_binders)) if list_of_binders else 0
	chunks = make_chunks(list_of_binders, num_threads) if num_threads > 0 else {}
	log(f"Num chunks: {chunks}")

	base_output = Path(cfg["directory_name"])
	run_dir = base_output / cfg["run_name"]
	scoring_dir = run_dir / "scoring"
	step1_dir = run_dir / "step1"
	step2_dir = run_dir / "step2"
	backrub_dir = run_dir / "backrub_outputs"

	for d in [base_output, run_dir, scoring_dir, step1_dir, step2_dir, backrub_dir]:
		make_dir(d)

	protein_base_name = os.path.basename(cfg["directory_name"])
	csv_suffix = f"_{protein_base_name}_{cfg['run_name']}.csv"

	csv_step1 = str(scoring_dir / f"step_1_design{csv_suffix}")
	csv_step2 = str(scoring_dir / f"step_2_design{csv_suffix}")

	threads = []
	for thread_num in range(num_threads):
		args_tuple = (
			chunks[thread_num],
			str(step1_dir),
			str(step2_dir),
			str(backrub_dir),
			thread_num,
			str(run_dir),
			cfg,
			csv_step1,
			csv_step2,
		)

		p = mp.Process(target=execute_pipeline, args=args_tuple)
		threads.append(p)
		p.start()

	for t in threads:
		t.join()
	
	merge_worker_csvs(csv_step1, num_threads, dedupe_cols=("name",))
	merge_worker_csvs(csv_step2, num_threads, dedupe_cols=("name",))

	log(f"All outputs stored in: {run_dir}\n")


if __name__ == "__main__":
	main()