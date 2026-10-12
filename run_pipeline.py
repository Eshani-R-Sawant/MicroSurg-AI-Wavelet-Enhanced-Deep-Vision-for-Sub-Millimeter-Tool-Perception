#!/usr/bin/env python3
"""
Root Pipeline Orchestrator for SurgMLLM / ZWZ.
Executes the end-to-end workflow:
1. Subsetting (data_pipeline.setup_and_subset)
2. SAM Crop Extraction (data_pipeline.micro_region_extractor)
3. VQA Cleaning (data_pipeline.cleanup_vqa)
4. Assembly (data_pipeline.dsyn_assembly)
"""

import argparse
from data_pipeline.setup_and_subset import create_subset
from data_pipeline.cleanup_vqa import run_repair
from data_pipeline.dsyn_assembly import assemble_training_dataset


def main():
    parser = argparse.ArgumentParser(description="Run ZWZ End-to-End Pipeline Stages")
    parser.add_argument("--stage", choices=["subset", "clean", "assemble", "all"], default="all")
    args = parser.parse_args()

    if args.stage in ["subset", "all"]:
        print("\n--- [Stage 1] Subsetting Surgical Datasets ---")
        create_subset()

    if args.stage in ["clean", "all"]:
        print("\n--- [Stage 2] Cleaning & Normalizing VQA Pairs ---")
        run_repair()

    if args.stage in ["assemble", "all"]:
        print("\n--- [Stage 3] Assembling Multimodal Dataset ---")
        assemble_training_dataset()

    print("\n✅ Pipeline stage(s) completed successfully.")


if __name__ == "__main__":
    main()
