"""Reproducible staged training and held-out ablation campaign."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch

from .data import Preprocessor, make_factorial, read_bpi, sample_batch, split_entities, stack_prefixes
from .evaluation import bootstrap_compare, entity_energy_scores, evaluate, export_frames
from .models import EventAE, PGGenerator


ROOT = Path(__file__).resolve().parents[1]
EVENT_KEYS = ("x_num", "x_cat", "mask", "delta", "t_valid")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def convert(x):
        if isinstance(x, (np.ndarray, torch.Tensor)):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x).__name__)
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=convert)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def source_hashes():
    files = list((ROOT / "pgjepa").glob("*.py")) + [ROOT / "vendor/tab_gen/score.py"]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def prepare(output, config, dataset="synthetic", raw_cache=None, bpi=None):
    output = Path(output)
    if dataset == "synthetic":
        entities = make_factorial(config["entities"], config["data_seed"])
        audit: dict[str, Any] = {"dataset": "controlled_16_cell_factorial", "time_unit": "days"}
    elif dataset == "retail":
        cached = torch.load(raw_cache, map_location="cpu")
        entities, audit = cached["entities"], cached["audit"]
    elif dataset == "bpi":
        entities, audit = read_bpi(bpi)
    else:
        raise ValueError(dataset)
    splits = split_entities(entities, seed=config["data_seed"])
    prep = Preprocessor().fit(splits["train"])
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "preprocessor.json", prep.state_dict())
    split_ids = {}
    for name, records in splits.items():
        encoded = [prep.transform(e) for e in records]
        torch.save(encoded, output / f"{name}.pt")
        split_ids[name] = [e["entity_id"] for e in records]
        audit[name] = {"entities": len(records), "events": sum(len(e["delta"]) for e in records),
                       "eligible": {str(k): sum(len(e["delta"]) >= config["prefix_test"] + k for e in records)
                                    for k in config["generation_lengths"]}}
        write_json(output / f"{name}_metadata.json", [{"entity_id": e["entity_id"],
                   "metadata": e.get("metadata", {})} for e in records])
    write_json(output / "split_ids.json", split_ids)
    write_json(output / "audit.json", audit)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob("*.pt")}
    write_json(output / "manifest.json", {"dataset": dataset, "config": config,
               "data_hashes": hashes, "preprocessor_hash": digest(prep.state_dict()),
               "source_hashes": source_hashes()})
    print(f"Prepared {dataset}: {audit}", flush=True)


def _load_split(data, name):
    manifest = read_json(Path(data) / "manifest.json")
    path = Path(data) / f"{name}.pt"
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["data_hashes"][path.name]:
        raise ValueError(f"Prepared data changed: {path}")
    return torch.load(path, map_location="cpu")


def _load_preprocessor(data):
    state = read_json(Path(data) / "preprocessor.json")
    if digest(state) != read_json(Path(data) / "manifest.json")["preprocessor_hash"]:
        raise ValueError("Train-fitted preprocessing changed")
    return Preprocessor.from_state_dict(state)


def _ae_rows(records, device):
    return {k: torch.cat([e[k] for e in records]).to(device) for k in EVENT_KEYS}


def _take_rows(rows, count, rng):
    indices = torch.as_tensor(rng.integers(len(rows["x_num"]), size=count), device=rows["x_num"].device)
    return {k: v[indices] for k, v in rows.items()}


def _rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def train_stage(model, stage, train, val, config, output, seed, device):
    """Finite fixed-budget stage; validation RNG is isolated from the train stream."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if stage == "A":
        for p in model.parameters():
            p.requires_grad_(True)
        rows, val_rows = _ae_rows(train, device), _ae_rows(val, device)
    else:
        model.configure_stage(stage)
    seed_all(seed)
    rng = np.random.default_rng(seed)
    steps = config[{"A": "ae_steps", "B": "jepa_steps", "C": "diffusion_steps_train"}[stage]]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                           lr=config["learning_rates"][stage])
    best, history, start = float("inf"), [], time.perf_counter()
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for step in range(1, steps + 1):
        model.train()
        if stage == "A":
            loss, stats = model.reconstruction_loss(_take_rows(rows, config["batch_size"] * 4, rng))
        else:
            batch, cut = sample_batch(train, config["batch_size"], tuple(config["prefix_range"]),
                                      max(config["horizons"]) if stage == "B" else 1, rng, device)
            loss, stats = (model.jepa_loss(batch, cut) if stage == "B" else model.generation_loss(batch, cut))
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Nonfinite {stage} objective at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
        if not bool(torch.isfinite(grad)):
            raise RuntimeError(f"Nonfinite {stage} gradient at step {step}")
        opt.step()
        if stage == "B" and model.arm != "A6":
            model.update_ema()
        if step == 1 or step % config["validation_interval"] == 0 or step == steps:
            state = _rng_state()
            seed_all(seed + 800000)
            vrng = np.random.default_rng(seed + 800000)
            validation = []
            diagnostics = []
            model.eval()
            with torch.no_grad():
                for _ in range(config["validation_batches"]):
                    if stage == "A":
                        vl, ds = model.reconstruction_loss(_take_rows(val_rows, config["batch_size"] * 4, vrng))
                    else:
                        vb, vc = sample_batch(val, config["batch_size"], tuple(config["prefix_range"]),
                                             max(config["horizons"]) if stage == "B" else 1, vrng, device)
                        vl, ds = model.jepa_loss(vb, vc) if stage == "B" else model.generation_loss(vb, vc)
                    validation.append(float(vl))
                    diagnostics.append(ds)
            _restore_rng(state)
            val_loss = float(np.mean(validation))
            merged = {k: float(np.mean([d[k] for d in diagnostics if k in d])) for k in diagnostics[0]}
            record = {"step": step, "train_loss": float(loss.detach()), "validation_loss": val_loss,
                      "diagnostics": merged, "elapsed_seconds": time.perf_counter() - start,
                      "gradient_norm": float(grad)}
            history.append(record)
            if val_loss < best:
                best = val_loss
                torch.save({"model": model.state_dict(), "stage": stage, "step": step,
                            "config": config, "seed": seed, "validation_loss": val_loss,
                            "diagnostics": merged, "rng": state, "optimizer": opt.state_dict()}, output / "best.pt")
            write_json(output / "history.json", history)
            print(f"{output} stage{stage} {step}/{steps} val={val_loss:.5f} elapsed={record['elapsed_seconds']:.1f}s", flush=True)
    checkpoint = torch.load(output / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    elapsed = time.perf_counter() - start
    result = {"stage": stage, "budget_steps": steps, "selected_step": checkpoint["step"],
              "validation_loss": checkpoint["validation_loss"], "diagnostics": checkpoint["diagnostics"],
              "elapsed_seconds": elapsed, "device": device,
              "peak_cuda_bytes": torch.cuda.max_memory_allocated() if device.startswith("cuda") else 0}
    write_json(output / "complete.json", result)
    return result


def build_generator(ae, prep, config, arm):
    cfg = config["model"]
    return PGGenerator(copy.deepcopy(ae), attrs_dim=prep.attrs_dim, arm=arm,
                       horizons=tuple(config["horizons"]), hidden=cfg["hidden"],
                       entity_dim=cfg["entity_dim"], width=cfg["width"], blocks=cfg["blocks"],
                       diffusion_steps=cfg["diffusion_steps"], zero_gap=True,
                       diffusion_beta_max=cfg.get("diffusion_beta_max", 0.1))


def numeric_domains(prep):
    return [{"count": "nonnegative_count", "distinct_products": "positive_count",
             "quantity": "signed_integer", "amount": "positive"}.get(name, "real")
            for name in prep.numeric_names]


def _gate_ae(stats, config):
    d, g = stats["diagnostics"], config["ae_gate"]
    return (d["numeric_mse"] <= g["numeric_mse_max"] and
            d["categorical_accuracy"] >= g["categorical_accuracy_min"] and
            d["mask_accuracy"] >= g["mask_accuracy_min"])


def _gate_jepa(stats, config):
    d, g = stats["diagnostics"], config["jepa_gate"]
    return all(d.get(f"target_std_{h}", 0) > g["target_std_min"] and
               d.get(f"prediction_std_{h}", 0) > g["prediction_std_min"] for h in config["horizons"])


def train_campaign(data, output, config, device):
    data, output = Path(data), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"config": config, "data_manifest_hash": digest(read_json(data / "manifest.json")),
                "source_hashes": source_hashes(), "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None}
    if (output / "manifest.json").exists() and read_json(output / "manifest.json") != manifest:
        raise ValueError("Output belongs to another config/source/data; choose a new directory")
    write_json(output / "manifest.json", manifest)
    # Test tensors are deliberately not opened anywhere in the training phase.
    train, val = _load_split(data, "train"), _load_split(data, "val")
    prep = _load_preprocessor(data)
    seed_all(config["seeds"][0])
    ae = EventAE(prep.n_num, prep.cat_sizes, config["model"]["latent_dim"]).to(device)
    ae.configure_numeric_domains(prep.mean.tolist(), prep.scale.tolist(), numeric_domains(prep))
    ae_dir = output / "AE"
    if (ae_dir / "complete.json").exists():
        ae.load_state_dict(torch.load(ae_dir / "best.pt", map_location=device)["model"])
        ae_stats = read_json(ae_dir / "complete.json")
    else:
        ae_stats = train_stage(ae, "A", train, val, config, ae_dir, config["seeds"][0], device)
        if _gate_ae(ae_stats, config):
            with torch.no_grad():
                ae.fit_latent_stats(_ae_rows(train, device))
            torch.save(ae.state_dict(), ae_dir / "frozen.pt")
    if not _gate_ae(ae_stats, config):
        write_json(output / "failed_gate.json", {"stage": "A", "statistics": ae_stats})
        raise RuntimeError("AE validation sanity gate failed; downstream experiment prohibited")
    ae.load_state_dict(torch.load(ae_dir / "frozen.pt", map_location=device))
    for seed in config["seeds"]:
        # Shared B checkpoint ensures A3/A4 differ only in generation conditioning.
        for b_arm in ("A4", "A6"):
            if not any(a in config["arms"] for a in (("A3", "A4") if b_arm == "A4" else ("A6",))):
                continue
            seed_all(seed)
            model = build_generator(ae, prep, config, b_arm).to(device)
            directory = output / f"seed_{seed}" / f"B_{b_arm}"
            if not (directory / "complete.json").exists():
                stats = train_stage(model, "B", train, val, config, directory, seed, device)
            else:
                stats = read_json(directory / "complete.json")
            if not _gate_jepa(stats, config):
                write_json(directory / "failed_gate.json", stats)
                raise RuntimeError(f"Stage B collapse gate failed: seed {seed}, {b_arm}")
        for arm in config["arms"]:
            directory = output / f"seed_{seed}" / arm
            if (directory / "complete.json").exists():
                continue
            seed_all(seed)
            model = build_generator(ae, prep, config, arm).to(device)
            if arm in {"A3", "A4", "A6"}:
                b_arm = "A6" if arm == "A6" else "A4"
                model.load_state_dict(torch.load(output / f"seed_{seed}" / f"B_{b_arm}" / "best.pt", map_location=device)["model"])
            stats = train_stage(model, "C", train, val, config, directory, seed, device)
            write_json(directory / "parameters.json", model.parameter_counts())
            print(f"Completed {arm} seed {seed}: {stats['validation_loss']:.5f}", flush=True)
    write_json(output / "training_complete.json", {"seeds": config["seeds"], "arms": config["arms"]})


def observation_arrays(batch, prep):
    result = {k: batch[k].detach().cpu().numpy() for k in ("x_num", "x_cat", "mask", "delta")}
    result["x_num"] = prep.inverse_numeric(result["x_num"])
    return result


def slice_events(data, length):
    return {k: v[..., :length, :] for k, v in data.items()}


def matched_horizon(real, generated, lengths, horizon):
    keep = np.asarray(lengths) >= horizon
    return (slice_events({k: v[keep] for k, v in real.items()}, horizon),
            slice_events({k: v[keep] for k, v in generated.items()}, horizon))


def _numeric_validity(generated, prep):
    n = generated["x_num"]
    mask = generated["mask"][..., :prep.n_num] > .5
    valid: dict[str, Any] = {"nonfinite_observed_rate": float((~np.isfinite(n) & mask).sum() / max(mask.sum(), 1)),
                             "features": {}}
    for j, name in enumerate(prep.numeric_names):
        x = n[..., j][mask[..., j]]
        d = {}
        if name in {"count", "distinct_products"}:
            d = {"negative_rate": float(np.mean(x < 0)),
                 "noninteger_rate": float(np.mean(np.abs(x - np.round(x)) > 1e-5))}
        elif name == "amount":
            d = {"negative_rate": float(np.mean(x < 0))}
        valid["features"][name] = d
    return valid


def evaluate_campaign(data, output, config, device, interventions=True):
    data, output = Path(data), Path(output)
    if not (output / "training_complete.json").exists():
        raise ValueError("Finish all training before test evaluation")
    training_manifest = read_json(output / "manifest.json")
    if training_manifest["config"] != config or training_manifest["data_manifest_hash"] != digest(read_json(data / "manifest.json")):
        raise ValueError("Evaluation config or data differ from the trained campaign")
    prep = _load_preprocessor(data)
    evaluation_manifest = {"config": config, "preprocessor_hash": digest(prep.state_dict()),
                           "metric_sources": {k: v for k, v in source_hashes().items()
                                              if k in {"pgjepa/evaluation.py", "vendor/tab_gen/score.py"}},
                           "data_manifest_hash": digest(read_json(data / "manifest.json"))}
    if (output / "evaluation_manifest.json").exists() and read_json(output / "evaluation_manifest.json") != evaluation_manifest and list(output.glob("seed_*/A*/evaluation_*/suffixes.npz")):
        raise ValueError("Cached metric definition changed; use a fresh evaluation directory")
    write_json(output / "evaluation_manifest.json", evaluation_manifest)
    test = _load_split(data, "test")
    maximum = max(config["generation_lengths"])
    minimum = min(config["generation_lengths"])
    eligible = [e for e in test if len(e["delta"]) >= config["prefix_test"] + minimum]
    if len(eligible) < 2:
        raise ValueError("Fewer than two held-out entities support longest horizon; use a separately labelled shorter-horizon config")
    prefix = stack_prefixes(eligible, config["prefix_test"], device)
    lengths = [min(maximum, len(e["delta"]) - config["prefix_test"]) for e in eligible]
    actual = {}
    for k in EVENT_KEYS:
        records = []
        for e, length in zip(eligible, lengths):
            values = e[k][config["prefix_test"]:config["prefix_test"] + length]
            padding = values.new_zeros((maximum - length, *values.shape[1:]))
            records.append(torch.cat([values, padding]))
        actual[k] = torch.stack(records)
    real = observation_arrays(actual, prep)
    np.savez_compressed(output / "real_suffixes.npz", **real)
    meta = read_json(data / "test_metadata.json")
    ids = read_json(data / "split_ids.json")["test"]
    chosen = [i for i, e in enumerate(test) if len(e["delta"]) >= config["prefix_test"] + minimum]
    write_json(output / "evaluation_protocol.json", {"eligible_entities": len(eligible), "total_test_entities": len(test),
               "entity_ids": [ids[i] for i in chosen], "metadata": [meta[i] for i in chosen],
               "prefix_events": config["prefix_test"], "generated_lengths": config["generation_lengths"],
               "replicates": config["replicates"], "cohort": "horizon-specific length-eligible matched entities",
               "observed_suffix_lengths": lengths,
               "support_by_horizon": {str(k): sum(v >= k for v in lengths) for k in config["generation_lengths"]},
               "saved_real_tensor_warning": "Right padding beyond observed_suffix_lengths is excluded before every metric; never score padded rows",
               "reference_commit": "3cd17110f0f5fe7ea25377a8e2db6a5840b622b5",
               "primary_score": "energy score on numeric mean/std, lag correlations, quarter-block shifts, categorical repeat rates, log duration and missingness",
               "primary_scale": "pooled real evaluation numeric std, common to every arm; never used for fitting the generator"})
    for seed in config["seeds"]:
        for arm in config["arms"]:
            run_dir = output / f"seed_{seed}" / arm
            seed_all(seed)
            ae = EventAE(prep.n_num, prep.cat_sizes, config["model"]["latent_dim"])
            model = build_generator(ae, prep, config, arm).to(device)
            model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device)["model"])
            model.eval()
            checkpoint_hash = hashlib.sha256((run_dir / "best.pt").read_bytes()).hexdigest()
            intervention_set = ["normal"]
            if arm == "A4" and interventions:
                intervention_set += ["zero_plan", "shuffle_plan", "direct_latent", "no_feedback"]
            for intervention in intervention_set:
                target = run_dir / f"evaluation_{intervention}"
                target.mkdir(exist_ok=True)
                sample_file = target / "suffixes.npz"
                fingerprint = digest({"checkpoint": checkpoint_hash, "config": config,
                                      "model_source": source_hashes()["pgjepa/models.py"],
                                      "preprocessing": digest(prep.state_dict()),
                                      "entity_ids": [ids[i] for i in chosen],
                                      "intervention": intervention, "seed": seed})
                if sample_file.exists():
                    if read_json(target / "sampling.json").get("fingerprint") != fingerprint:
                        raise ValueError(f"Stale generated sample cache: {sample_file}")
                    with np.load(sample_file) as arrays:
                        generated = {k: arrays[k] for k in arrays.files}
                else:
                    samples = []
                    started = time.perf_counter()
                    model.ae.reset_decoder_audit()
                    for replicate in range(config["replicates"]):
                        seed_all(seed + 100000 + replicate)
                        # Stable full-cohort intervention swaps; no original future is passed.
                        samples.append(observation_arrays(model.generate(prefix, maximum, config["sample_steps"], intervention), prep))
                    generated = {k: np.stack([s[k] for s in samples], axis=1) for k in real}
                    np.savez_compressed(sample_file, **generated)
                    write_json(target / "sampling.json", {"fingerprint": fingerprint,
                               "checkpoint_sha256": checkpoint_hash,
                               "wall_seconds": time.perf_counter() - started,
                               "numeric_validity": _numeric_validity(generated, prep),
                               "decoder_domain_policy": model.ae.decoder_audit(),
                               "diffusion_beta_max": config["model"].get("diffusion_beta_max", .1),
                               "terminal_alpha_bar": float(model.alpha_bar[-1])})
                for length in config["generation_lengths"]:
                    metric_file = target / f"metrics_K{length}.json"
                    if metric_file.exists():
                        continue
                    r, g = matched_horizon(real, generated, lengths, length)
                    if len(r["x_num"]) < 2:
                        write_json(metric_file, {"status": "insufficient_entities", "ts": None,
                                   "cs": None, "sd": None, "lg": None, "primary_energy_score": None})
                        continue
                    metrics = evaluate(r, g, dataset="pg_" + read_json(data / "manifest.json")["dataset"])
                    np.save(target / f"entity_scores_K{length}.npy", entity_energy_scores(r, g))
                    write_json(metric_file, metrics)
                    print(f"Evaluated seed{seed} {arm} {intervention} K{length}: ts={metrics['ts']} sd={metrics['sd']} lg={metrics['lg']} ES={metrics['primary_energy_score']}", flush=True)
                if arm == "A4" and intervention == "normal":
                    rf, sf = export_frames(*matched_horizon(real, generated, lengths, maximum))
                    rf.to_csv(output / "real_reference.csv", index=False)
                    sf.to_csv(target / "synthetic_reference.csv", index=False)
    comparisons = []
    for seed in config["seeds"]:
        for length in config["generation_lengths"]:
            for baseline in ("A2", "A3", "A6"):
                if baseline not in config["arms"] or "A4" not in config["arms"]:
                    continue
                with np.load(output / f"seed_{seed}" / baseline / "evaluation_normal/suffixes.npz") as a, np.load(output / f"seed_{seed}" / "A4/evaluation_normal/suffixes.npz") as b:
                    r, ga = matched_horizon(real, dict(a), lengths, length)
                    _, gb = matched_horizon(real, dict(b), lengths, length)
                    if len(r["x_num"]) < 2:
                        continue
                    cmp = bootstrap_compare(r, ga, gb, config["bootstrap_resamples"], seed)
                cmp.update(seed=seed, length=length, baseline=baseline, proposed="A4")
                comparisons.append(cmp)
    write_json(output / "comparisons.json", comparisons)
    build_report(output, config)


def build_report(output, config):
    output = Path(output)
    protocol = read_json(output / "evaluation_protocol.json")
    lines = ["# PG-JEPA v0.2 experiment results", "", f"Held-out cohort: {protocol['eligible_entities']} / {protocol['total_test_entities']} entities; support by K={protocol['support_by_horizon']}; P={config['prefix_test']}; S={config['replicates']}; seeds={config['seeds']}.",
             "", "Reference TS/CS/SD/LG directly use pinned tab_gen eval/score.py. Higher is better. Conditional summary Energy Score (ES) is supplemental, lower is better. K32/64 are extrapolative. No closed-loop training or LGCP was performed.",
             "", "| Arm | K | TS | CS | SD | LG | ES |", "|---|---:|---:|---:|---:|---:|---:|"]
    rows = []
    for arm in config["arms"]:
        for k in config["generation_lengths"]:
            ds = [read_json(output / f"seed_{s}" / arm / f"evaluation_normal/metrics_K{k}.json") for s in config["seeds"]]
            vals = []
            for metric in ("ts", "cs", "sd", "lg", "primary_energy_score"):
                numbers = [d[metric] for d in ds if d[metric] is not None]
                vals.append(f"{np.mean(numbers):.4f} ± {np.std(numbers, ddof=1) if len(numbers)>1 else 0:.4f}" if numbers else "unsupported")
            lines.append(f"| {arm} | {k} | " + " | ".join(vals) + " |")
            rows.append({"arm": arm, "K": k, **{metric: [d[metric] for d in ds] for metric in ("ts", "cs", "sd", "lg", "primary_energy_score")}})
    lines += ["", "Paired per-seed entity-bootstrap effects and confidence intervals are in comparisons.json. Three training seeds quantify initialization variability; per-seed entity CIs alone do not establish across-seed significance.",
              "", "Parameter counts, stage budgets, validation-selected steps and compute are stored in each run directory. Numeric-domain violations, finite metric coverage, sparse daily-grid behavior and missingness must be considered alongside headline scores."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(output / "summary.json", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "train", "evaluate", "run"])
    parser.add_argument("--config", default=str(ROOT / "configs/pg_jepa_v0.2.json"))
    parser.add_argument("--data", default="data/PG_Synthetic")
    parser.add_argument("--output", default="outputs/pg_jepa_v02")
    parser.add_argument("--dataset", choices=["synthetic", "retail", "bpi"], default="synthetic")
    parser.add_argument("--raw-cache", default="data/PG_RetailII/entities.pt")
    parser.add_argument("--bpi")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = read_json(args.config)
    if args.command in {"prepare", "run"} and not (Path(args.data) / "manifest.json").exists():
        prepare(args.data, config, args.dataset, args.raw_cache, args.bpi)
    if args.command in {"train", "run"}:
        train_campaign(args.data, args.output, config, args.device)
    if args.command in {"evaluate", "run"}:
        evaluate_campaign(args.data, args.output, config, args.device)


if __name__ == "__main__":
    main()
