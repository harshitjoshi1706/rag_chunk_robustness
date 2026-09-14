"""Sequential, resumable OHRBench chunking robustness experiment.

Use `OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` on this Mac.
Each configuration runs in a fresh process so its memory high-water is isolated.
"""

import argparse
import csv
import gc
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from chunkers import (DEFAULT_FIXED_CHUNK_SIZE, DEFAULT_FIXED_CHUNK_OVERLAP,
                      DEFAULT_SEMANTIC_THRESHOLD, DEFAULT_SEMANTIC_MIN_TOKENS,
                      DEFAULT_SEMANTIC_MAX_TOKENS, DEFAULT_STRUCTURE_MAX_TOKENS)
from embeddings import MODEL_NAME, embed_chunk_texts, load_embedding_model
from faiss_index import add_vectors, create_index, load_index, save_index
from official_qa import load_mapped_qas
from ohrbench_adapter import TEXT_COLUMNS, iter_documents, load_pages
from retrieval_metrics import score_ranking
from retrieval_pipeline import CHUNKERS, chunk_document, retrieve_one


RESULTS = ROOT / "results"
INDEXES = ROOT / "indexes"
RUN_ID = "ohrbench_minilm_254_v1"
TOP_K = 100
QUERY_BATCH = 128
EMBED_BATCH = 32
QUERY_COLUMNS = ["run_id", "noise_level", "chunker", "query_id", "doc_name", "domain",
                 "context_verified", "hit_at_1", "hit_at_5", "reciprocal_rank",
                 "first_relevant_rank", "retrieval_latency_ms"]
CONFIG_COLUMNS = ["run_id", "noise_level", "chunker", "num_documents", "num_pages",
                  "num_chunks", "mean_chunk_tokens", "median_chunk_tokens", "std_chunk_tokens",
                  "min_chunk_tokens", "max_chunk_tokens", "chunking_time_sec", "embedding_time_sec",
                  "index_build_time_sec", "query_embedding_time_sec", "total_retrieval_time_sec",
                  "mean_query_latency_ms", "median_query_latency_ms", "peak_memory_mb",
                  "index_size_bytes", "status"]


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_qas():
    clean = load_pages("clean")[["doc_name", "page_idx", "gt_text"]]
    qas, excluded = load_mapped_qas(clean)
    qas.sort(key=lambda item: item["query_id"])
    if len(qas) != 8456 or sum(q["context_verified"] for q in qas) != 5277:
        raise RuntimeError("Phase 7B QA counts changed")
    return qas, excluded


def prepare_queries(qas, model=None):
    path = INDEXES / "official_query_vectors.npy"
    ids_path = INDEXES / "official_query_ids.json"
    ids = [qa["query_id"] for qa in qas]
    if path.exists() and ids_path.exists():
        if json.loads(ids_path.read_text(encoding="utf-8")) != ids:
            raise RuntimeError("Cached query IDs do not match official QAs")
        vectors = np.load(path, mmap_mode="r")
        if vectors.shape != (8456, 384):
            raise RuntimeError("Cached query embedding shape changed")
        return vectors, 0.0
    if model is None:
        model = load_embedding_model()
    t0 = time.perf_counter()
    vectors = np.empty((len(qas), 384), dtype=np.float32)
    for start in range(0, len(qas), EMBED_BATCH):
        texts = [q["question"] for q in qas[start:start + EMBED_BATCH]]
        vectors[start:start + len(texts)] = embed_chunk_texts(texts, model, EMBED_BATCH)
    elapsed = time.perf_counter() - t0
    INDEXES.mkdir(exist_ok=True)
    temporary = path.with_suffix(".npy.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, vectors)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    atomic_json(ids_path, ids)
    return np.load(path, mmap_mode="r"), elapsed


def configuration_key(chunker, noise):
    return f"{chunker}_{noise}"


def paths_for(key):
    return {
        "index": INDEXES / f"{key}.faiss",
        "metadata": INDEXES / f"{key}.jsonl",
        "ready": INDEXES / f"{key}.ready.json",
        "profile": RESULTS / "profiles" / f"{key}.json",
        "parts": RESULTS / "query_parts" / key,
    }


def build_configuration(chunker, noise, paths, model):
    pages = load_pages(noise)
    tokenizer = model.tokenizer
    index = create_index()
    metadata = []
    lengths = []
    chunking_sec = embedding_sec = index_sec = 0.0
    temporary_metadata = paths["metadata"].with_suffix(".jsonl.tmp")
    with temporary_metadata.open("w", encoding="utf-8") as metadata_stream:
        for number, document in enumerate(iter_documents(pages, noise), 1):
            t0 = time.perf_counter()
            chunks = chunk_document(document, chunker, tokenizer, model)
            chunking_sec += time.perf_counter() - t0
            for chunk in chunks:
                entry = {"document_id": chunk["document_id"], "source_page_ids": chunk["source_page_ids"],
                         "chunker": chunker, "noise_level": noise, "token_count": chunk["token_count"],
                         "text_ref": {"doc_name": chunk["document_id"], "start_char": chunk["start_char"],
                                      "end_char": chunk["end_char"]}}
                metadata.append(entry)
                lengths.append(chunk["token_count"])
                metadata_stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
            for start in range(0, len(chunks), EMBED_BATCH):
                texts = [chunk["text"] for chunk in chunks[start:start + EMBED_BATCH]]
                t0 = time.perf_counter()
                vectors = embed_chunk_texts(texts, model, EMBED_BATCH)
                embedding_sec += time.perf_counter() - t0
                t0 = time.perf_counter()
                add_vectors(index, vectors)
                index_sec += time.perf_counter() - t0
            if number % 50 == 0 or number == 1261:
                print(f"{chunker}/{noise}: documents {number}/1261, chunks {index.ntotal}", flush=True)
            del chunks
        metadata_stream.flush()
        os.fsync(metadata_stream.fileno())
    if index.ntotal != len(metadata):
        raise RuntimeError("FAISS position-to-metadata count mismatch")
    temporary_index = paths["index"].with_suffix(".faiss.tmp")
    save_index(index, temporary_index)
    os.replace(temporary_index, paths["index"])
    os.replace(temporary_metadata, paths["metadata"])
    atomic_json(paths["ready"], {"ntotal": index.ntotal, "dimension": index.d,
                                 "run_id": RUN_ID, "chunker": chunker, "noise_level": noise})
    values = np.asarray(lengths, dtype=np.float32)
    profile = {"run_id": RUN_ID, "noise_level": noise, "chunker": chunker,
               "num_documents": int(pages["doc_name"].nunique()), "num_pages": len(pages),
               "num_chunks": index.ntotal, "mean_chunk_tokens": float(values.mean()),
               "median_chunk_tokens": float(np.median(values)), "std_chunk_tokens": float(values.std()),
               "min_chunk_tokens": int(values.min()), "max_chunk_tokens": int(values.max()),
               "chunking_time_sec": chunking_sec, "embedding_time_sec": embedding_sec,
               "index_build_time_sec": index_sec, "query_embedding_time_sec": 0.0,
               "index_size_bytes": paths["index"].stat().st_size}
    atomic_json(paths["profile"], profile)
    del pages, lengths, values
    gc.collect()
    return index, metadata, profile


def load_built_configuration(paths):
    ready = json.loads(paths["ready"].read_text(encoding="utf-8"))
    if ready["run_id"] != RUN_ID:
        raise RuntimeError("Index run ID mismatch")
    index = load_index(paths["index"])
    with paths["metadata"].open(encoding="utf-8") as stream:
        metadata = [json.loads(line) for line in stream]
    if index.ntotal != ready["ntotal"] or len(metadata) != index.ntotal:
        raise RuntimeError("Persisted index/metadata mismatch")
    profile = json.loads(paths["profile"].read_text(encoding="utf-8"))
    return index, metadata, profile


def completed_query_ids(parts_dir):
    done = set()
    for path in sorted(parts_dir.glob("batch_*.csv")):
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row["query_id"] in done:
                    raise RuntimeError(f"Duplicate query result: {row['query_id']}")
                done.add(row["query_id"])
    return done


def run_configuration(chunker, noise):
    key = configuration_key(chunker, noise)
    paths = paths_for(key)
    paths["parts"].mkdir(parents=True, exist_ok=True)
    qas, _ = read_qas()
    query_vectors, _ = prepare_queries(qas)
    if all(paths[name].exists() for name in ("index", "metadata", "ready", "profile")):
        index, metadata, profile = load_built_configuration(paths)
        print(f"Resumed existing index {key}: {index.ntotal} vectors", flush=True)
    else:
        model = load_embedding_model()
        index, metadata, profile = build_configuration(chunker, noise, paths, model)
        del model
        gc.collect()
    done = completed_query_ids(paths["parts"])
    if not done.issubset({q["query_id"] for q in qas}):
        raise RuntimeError("Saved query results contain unknown IDs")
    for start in range(0, len(qas), QUERY_BATCH):
        batch = qas[start:start + QUERY_BATCH]
        path = paths["parts"] / f"batch_{start:06d}.csv"
        if path.exists():
            continue
        rows = []
        for offset, qa in enumerate(batch):
            t0 = time.perf_counter()
            ranking = retrieve_one(index, metadata, query_vectors[start + offset], TOP_K)
            metrics = score_ranking([item for _, item in ranking], qa)
            elapsed = (time.perf_counter() - t0) * 1000
            rows.append({"run_id": RUN_ID, "noise_level": noise, "chunker": chunker,
                         "query_id": qa["query_id"], "doc_name": qa["doc_name"],
                         "domain": qa["domain"], "context_verified": int(qa["context_verified"]),
                         **metrics, "retrieval_latency_ms": elapsed})
        atomic_csv(path, QUERY_COLUMNS, rows)
        if (start // QUERY_BATCH) % 10 == 0:
            print(f"{key}: searched {start + len(batch)}/{len(qas)} queries", flush=True)
    done = completed_query_ids(paths["parts"])
    if len(done) != 8456:
        raise RuntimeError(f"Incomplete query results for {key}: {len(done)}")
    latencies = []
    for path in sorted(paths["parts"].glob("batch_*.csv")):
        with path.open(newline="", encoding="utf-8") as stream:
            latencies.extend(float(row["retrieval_latency_ms"]) for row in csv.DictReader(stream))
    profile.update({"total_retrieval_time_sec": sum(latencies) / 1000,
                    "mean_query_latency_ms": float(np.mean(latencies)),
                    "median_query_latency_ms": float(np.median(latencies)),
                    "peak_memory_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2),
                    "status": "complete"})
    atomic_json(paths["profile"], profile)
    print(f"COMPLETE {key}: {index.ntotal} vectors", flush=True)


def aggregate_completed():
    profiles = []
    for chunker in CHUNKERS:
        for noise in TEXT_COLUMNS:
            path = paths_for(configuration_key(chunker, noise))["profile"]
            if path.exists():
                profile = json.loads(path.read_text(encoding="utf-8"))
                if profile.get("status") == "complete":
                    profiles.append(profile)
    atomic_csv(RESULTS / "configuration_results.csv", CONFIG_COLUMNS, profiles)
    output = RESULTS / "query_results.csv"
    temporary = output.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=QUERY_COLUMNS)
        writer.writeheader()
        for profile in profiles:
            parts = paths_for(configuration_key(profile["chunker"], profile["noise_level"]))["parts"]
            for part in sorted(parts.glob("batch_*.csv")):
                with part.open(newline="", encoding="utf-8") as source:
                    writer.writerows(csv.DictReader(source))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output)
    return profiles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs=2, metavar=("CHUNKER", "NOISE"))
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    INDEXES.mkdir(exist_ok=True)
    if args.config:
        chunker, noise = args.config
        if chunker not in CHUNKERS or noise not in TEXT_COLUMNS:
            parser.error("Unknown configuration")
        run_configuration(chunker, noise)
        return
    qas, excluded = read_qas()
    atomic_csv(RESULTS / "qa_mapping_report.csv", ["query_id", "doc_name", "reason"], excluded)
    qa_digest = hashlib.sha256((ROOT / "data/raw/qas_v2.json").read_bytes()).hexdigest()
    model = load_embedding_model()
    special = model.tokenizer.num_special_tokens_to_add(pair=False)
    if model.max_seq_length != 256 or special != 2:
        raise RuntimeError("Embedding token-budget preflight changed")
    _, query_time = prepare_queries(qas, model)
    del model
    gc.collect()
    manifest = {"run_id": RUN_ID, "model": MODEL_NAME, "device": "cpu", "embedding_dimension": 384,
                "model_max_seq_length": 256, "required_special_tokens": 2, "content_token_budget": 254,
                "token_budget_correction": "Uniform 256 to 254 content-token maximum before retrieval",
                "query_embedding_time_sec_this_invocation": query_time, "qa_sha256": qa_digest,
                "official_qas": 8498, "primary_qas": 8456, "excluded_absent_document": 42,
                "context_verified_qas": 5277, "top_k": TOP_K, "mrr_definition": "MRR@100",
                "chunker_parameters": {"fixed": {"size": DEFAULT_FIXED_CHUNK_SIZE,
                                                 "overlap": DEFAULT_FIXED_CHUNK_OVERLAP},
                                       "recursive": {"maximum": DEFAULT_FIXED_CHUNK_SIZE,
                                                     "overlap": DEFAULT_FIXED_CHUNK_OVERLAP},
                                       "semantic": {"threshold": DEFAULT_SEMANTIC_THRESHOLD,
                                                    "minimum": DEFAULT_SEMANTIC_MIN_TOKENS,
                                                    "maximum": DEFAULT_SEMANTIC_MAX_TOKENS},
                                       "structure": {"maximum": DEFAULT_STRUCTURE_MAX_TOKENS}},
                "started_at_epoch_sec": time.time(), "configurations": {}}
    previous = RESULTS / "experiment_manifest.json"
    if previous.exists():
        old = json.loads(previous.read_text(encoding="utf-8"))
        if old.get("qa_sha256") != qa_digest or old.get("run_id") != RUN_ID:
            raise RuntimeError("Existing manifest is for a different experiment")
        manifest = old
    for chunker in CHUNKERS:
        for noise in TEXT_COLUMNS:
            key = configuration_key(chunker, noise)
            profile = paths_for(key)["profile"]
            if profile.exists() and json.loads(profile.read_text(encoding="utf-8")).get("status") == "complete":
                manifest["configurations"][key] = "complete"
                continue
            manifest["configurations"][key] = "running"
            atomic_json(previous, manifest)
            command = [sys.executable, str(ROOT / "run_experiment.py"), "--config", chunker, noise]
            with (RESULTS / "experiment.log").open("a", encoding="utf-8") as log:
                log.write(f"START {time.time()} {key}\n")
                log.flush()
                completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                log.write(f"END {time.time()} {key} exit={completed.returncode}\n")
            if completed.returncode:
                manifest["configurations"][key] = f"failed_exit_{completed.returncode}"
                atomic_json(previous, manifest)
                aggregate_completed()
                raise RuntimeError(f"Configuration {key} failed; see results/experiment.log")
            manifest["configurations"][key] = "complete"
            atomic_json(previous, manifest)
            aggregate_completed()
    manifest["completed_at_epoch_sec"] = time.time()
    atomic_json(previous, manifest)
    aggregate_completed()


if __name__ == "__main__":
    main()
