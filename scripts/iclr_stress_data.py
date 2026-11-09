"""Development-only acquisition controls and procedural ODE scaling realizations."""

from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.benchmarks.phase_b_generation import PhaseBProtocol, _simulate_alien
from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.reference_integration import ReferenceSolver
from autoformalism.schemas.public_fitting import PublicSplit

T2 = "phase_c_dalla_man_t2_canonical_named_hard_rates_v1"
CSTR = "phase_c_cstr_controlled_reactor_mechanism_canonical_named_{}_reset_v1"
ALIEN = (
    "phase_c_alien_device_unknown_device_mechanism_canonical_functional_hard_reset_v1"
)
ORIGINAL = "phase_b_cstr_controlled_reactor_mechanism_canonical_named_easy"
CORRECTED = "cstr_easy_reference_events_v2"
CORRECTED_HASHES = {
    "train.csv": "112799ce2be91a95486b0efa2090782264a34f960fec94a6f69fc1175ca3d979",
    "validation.csv": (
        "b98d9950047068d952c3af17435b0fa8d3d6c44ff36779c15afba983345f3ef9"
    ),
    "proposer_prompt.txt": (
        "0a3b4c055402ed26352363c4d367b791c41460ea21e8ca42043c495ea68eb99c"
    ),
    "judge_prompt.txt": (
        "e71fc4a51abf0e6ba9033b02e45b6021b55d783f724cb723f3b1f2bb9ba61f3b"
    ),
}
NOISE_SEED = 72027
DRAW_SEEDS = (11639, 27181)
SOLVER = ReferenceSolver(method="DOP853", max_step=0.4, rtol=1e-9, atol=1e-11)
ROLES = ("targets", "auxiliaries", "external_inputs")


def digest(path: Path) -> str:
    """Return the digest of the exact input bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_cell(root: Path, key: str, spec: dict, train: list, val: list) -> dict:
    """Publish a Phase C-compatible search cell with verified role/array alignment."""
    cell = root / "public" / key
    for split, rows in (("train", train), ("val", val)):
        for r in rows:
            if any(set(r[role]) != set(spec[role]) for role in ROLES):
                raise ValueError("split and specification channel roles differ")
        data = PublicSplit.model_validate(
            {"name": split, "rows": rows, "fingerprint": content_sha256(rows)}
        )
        seal(cell / f"{split}.json", data.model_dump(mode="json"))
    seal(cell / "specification.json", {**spec, "benchmark_id": key})
    path = cell / "proposer_prompt.txt"
    prompt = spec["public_prompt"].rstrip("\n") + "\n"
    if path.exists() and path.read_text() != prompt:
        raise ValueError("prompt differs from prior preparation")
    path.write_text(prompt)
    return {
        "cell": key,
        "public_spec_sha256": digest(cell / "specification.json"),
        "public_files": {p.name: digest(p) for p in sorted(cell.iterdir())},
    }


def train_scales(rows: list[dict]) -> dict[str, float]:
    """Calculate population SD on clean training measurements."""
    channels = {}
    for role in ("targets", "auxiliaries"):
        for name in rows[0][role]:
            values = np.concatenate([r[role][name] for r in rows])
            channels[name] = 0.0 if np.ptp(values) == 0 else float(values.std())
    return channels


def acquire(
    row: dict,
    *,
    fraction: float = 0,
    stride: int = 1,
    scales: dict,
    mask: tuple | None = None,
) -> dict:
    """Preserve input corners and shared noise draws on the acquisition time grid."""
    t = np.asarray(row["time"])
    indices = set(range(0, len(t), stride)) | {len(t) - 1}
    for values in row["external_inputs"].values():
        slopes = np.diff(values) / np.diff(t)
        tol = 1e-12 * max(1, np.max(abs(slopes)))
        indices.update(1 + np.flatnonzero(abs(np.diff(slopes)) > tol))
    ids = sorted(indices)
    result = {
        "trajectory_id": row["trajectory_id"],
        "time": t[ids].tolist(),
        "fixed_covariates": row.get("fixed_covariates", {}),
    }
    for role in ROLES:
        channels = {}
        for name, v in row[role].items():
            if role == "auxiliaries" and mask is not None and name not in mask:
                continue
            values = np.asarray(v, dtype=float)
            if role == "external_inputs":
                if not np.allclose(
                    np.interp(t, t[ids], values[ids]), values, rtol=1e-11, atol=1e-12
                ):
                    raise ValueError("acquisition grid would change supplied forcing")
            elif fraction:
                seed = int(
                    content_sha256([NOISE_SEED, row["trajectory_id"], role, name])[:16],
                    16,
                )
                values = (
                    values
                    + np.random.default_rng(seed).normal(size=len(t))
                    * scales[name]
                    * fraction
                )
            channels[name] = values[ids].tolist()
        result[role] = channels
    return result


def accuracy_conditions(release: Path, root: Path) -> list[dict]:
    """Build T2 acquisition and CSTR mask cells from clean development data."""
    t2 = release / "public" / T2
    spec = read_seal(t2 / "specification.json")
    t2_rows = [read_seal(t2 / f"{split}.json")["rows"] for split in ("train", "val")]
    scales = train_scales(t2_rows[0])
    cells = []
    for i, (noise, stride) in enumerate(
        ((0, 1), (0.01, 1), (0.05, 1), (0.10, 1), (0, 2), (0, 4))
    ):
        rows = [
            [acquire(r, fraction=noise, stride=stride, scales=scales) for r in split]
            for split in t2_rows
        ]
        cell = write_cell(
            root,
            f"acquisition_{i:02}",
            {**spec, "noise_sd_fraction": noise, "initial_observation_noise": noise},
            *rows,
        )
        cells.append(
            {
                **cell,
                "study": "data_quality",
                "family": "t2",
                "source": T2,
                "noise": noise,
                "stride": stride,
                "scales": scales,
                "mask": ["Uii"],
                "train_samples": sum(len(r["time"]) for r in rows[0]),
            }
        )
    hard = release / "public" / CSTR.format("hard")
    easy = release / "public" / CSTR.format("easy")
    hard_spec = read_seal(hard / "specification.json")
    rows = []
    for split in ("train", "val"):
        rich = read_seal(easy / f"{split}.json")["rows"]
        for a, b in zip(rich, read_seal(hard / f"{split}.json")["rows"], strict=True):
            for k in ("trajectory_id", "time", "targets", "external_inputs"):
                if a[k] != b[k]:
                    raise ValueError(
                        "CSTR tiers contain different physical preparations"
                    )
        rows.append(rich)
    for i, mask in enumerate(((), ("C",), ("Tj",), ("C", "Tj"))):
        masked = [[acquire(r, scales={}, mask=mask) for r in split] for split in rows]
        desc = {
            "C": "- C(t): reactant concentration [unit: relative concentration]",
            "Tj": "- Tj(t): jacket temperature [unit: K]",
        }
        marker = "Supplied auxiliaries:\n- none\n"
        if hard_spec["public_prompt"].count(marker) != 1:
            raise ValueError("unexpected CSTR-hard acquisition description")
        prompt = hard_spec["public_prompt"].replace(
            marker,
            "Supplied auxiliaries:\n"
            + ("\n".join(desc[c] for c in mask) or "- none")
            + "\n",
        )
        cell = write_cell(
            root,
            f"sensor_{i:02}",
            {**hard_spec, "auxiliaries": list(mask), "public_prompt": prompt},
            *masked,
        )
        cells.append(
            {
                **cell,
                "study": "data_quality",
                "family": "cstr",
                "source": CSTR.format("hard"),
                "mask": list(mask),
                "noise": 0,
                "stride": 1,
            }
        )
    return cells


def corrected_cell(directory: Path, root: Path) -> dict:
    """Load the corrected historical export and its submitted easy prompt."""
    if {
        name: digest(directory / name) for name in CORRECTED_HASHES
    } != CORRECTED_HASHES:
        raise ValueError("expected reference-events-2 corrected historical CSTR export")
    splits = []
    for name in ("train", "validation"):
        grouped = defaultdict(list)
        with (directory / f"{name}.csv").open() as stream:
            for r in csv.DictReader(stream):
                grouped[r["trajectory_id"]].append(r)
        rows = []
        for key, group in grouped.items():

            def series(field: str, group: list = group) -> list:
                return [float(r[field]) for r in group]

            rows.append(
                {
                    "trajectory_id": key,
                    "time": series("t"),
                    "targets": {"T": series("T")},
                    "auxiliaries": {c: series(c) for c in ("C", "Tj")},
                    "external_inputs": {c: series(c) for c in ("Cf", "Tf", "Tjf")},
                }
            )
        splits.append(rows)
    spec = {
        "protocol": "iclr-corrected-original-cstr-1",
        "targets": ["T"],
        "auxiliaries": ["C", "Tj"],
        "fixed_covariates": [],
        "external_inputs": ["Cf", "Tf", "Tjf"],
        "input_contract": "legacy-events-1",
        "public_prompt": (directory / "proposer_prompt.txt").read_text().rstrip("\n"),
        "judge_prompt": (directory / "judge_prompt.txt").read_text().rstrip("\n"),
        "test_released": False,
    }
    return {
        **write_cell(root, CORRECTED, spec, *splits),
        "study": "corrected_easy",
        "family": "cstr",
        "source": ORIGINAL,
        "source_files": CORRECTED_HASHES,
        "mask": ["C", "Tj"],
        "noise": 0,
        "stride": 1,
    }


def sample_system(seed: int, index: int, n: int) -> dict:
    """Reproduce the trusted notebook's sample_candidate draw order and grammar."""
    seed += 1_000_003 * index
    rng = np.random.default_rng(seed)
    decay = rng.uniform(0.06, 0.18, n)
    skew = np.zeros((n, n))

    def edge(i: int, j: int, low: float, high: float) -> None:
        weight = rng.uniform(low, high) * rng.choice([-1.0, 1.0])
        skew[i, j] += weight
        skew[j, i] -= weight

    for i in range(n):
        edge(i, (i + 1) % n, 0.15, 0.50)
    for _ in range(2):
        i, j = rng.choice(n, 2, replace=False)
        edge(i, j, 0.10, 0.40)
    tanh, products = [], []
    for _ in range(n):
        tanh.append(
            [
                {
                    "source": int(rng.integers(n)),
                    "coefficient": rng.uniform(-0.70, 0.70),
                    "scale": rng.uniform(0.50, 1.80),
                    "bias": rng.uniform(-0.40, 0.40),
                }
                for _ in range(rng.integers(1, 3))
            ]
        )
        terms = []
        for _ in range(rng.integers(0, 2)):
            j, k = rng.choice(n, 2, replace=False)
            terms.append(
                {
                    "source_1": int(j),
                    "source_2": int(k),
                    "coefficient": rng.uniform(-0.45, 0.45),
                    "scale_1": rng.uniform(0.50, 1.50),
                    "scale_2": rng.uniform(0.50, 1.50),
                }
            )
        products.append(terms)
    b = np.zeros(n)
    for j in rng.choice(n, 2, replace=False):
        b[j] = rng.uniform(0.50, 1.20) * rng.choice([-1.0, 1.0])
    out = [
        {
            "source": int(j),
            "coefficient": rng.uniform(-1, 1),
            "scale": rng.uniform(0.50, 1.50),
        }
        for j in rng.choice(n, 3, replace=False)
    ]
    j, k = rng.choice(n, 2, replace=False)
    pair = [
        {
            "source_1": int(j),
            "source_2": int(k),
            "coefficient": rng.uniform(-0.60, 0.60),
            "scale_1": rng.uniform(0.50, 1.50),
            "scale_2": rng.uniform(0.50, 1.50),
        }
    ]
    return {
        "grammar_version": "alien_grammar_v1",
        "candidate_index": index,
        "candidate_seed": seed,
        "n_latent": n,
        "decay": decay.tolist(),
        "skew": skew.tolist(),
        "tanh_terms": tanh,
        "product_terms": products,
        "input_vector": b.tolist(),
        "input_scale": rng.uniform(0.70, 1.50),
        "output_decay": rng.uniform(0.08, 0.20),
        "output_terms": out,
        "output_product_terms": pair,
    }


def device_protocols(split: str) -> list[PhaseBProtocol]:
    """Use the same input schedules and units at every dimension."""
    designs = {
        "train": [
            [],
            [(3, 7, 1)],
            [(4, 9, -1)],
            [(3, 15, 0.6)],
            [(6, 10, 0.5), (15, 21, -0.7)],
            [(2, 8, -0.4), (12, 19, 0.8)],
        ],
        "validation": [[(4, 9, 0.8)], [(7, 12, -0.6)]],
        "test": [[], [(5, 13, 0.9)], [(3, 6, -0.7), (12, 20, 0.5)]],
    }
    return [
        PhaseBProtocol(
            protocol_id=f"{split}_pulse_{i}",
            family="alien_device",
            split=split,
            duration=24,
            dt=0.25,
            input_dt=0.25,
            input_names=("u",),
            input_contract="continuous-rates-1",
            specification={"kind": "pulses", "pulses": p},
        )
        for i, p in enumerate(designs[split])
    ]


def device_row(tr, index: int, split: str) -> dict:
    return {
        "trajectory_id": f"{split}_{index:03d}",
        "time": tr.time.tolist(),
        "targets": {"v01": tr.states[:, -1].tolist()},
        "auxiliaries": {},
        "external_inputs": {"u01": tr.inputs[:, 0].tolist()},
    }


def qualify(spec: dict) -> dict:
    """Check training excitation and the output influence of every hidden coordinate."""
    quiet, pulse = device_protocols("train")[:2]

    def response(prot, offsets: dict) -> np.ndarray:
        return _simulate_alien(prot, spec, offsets, SOLVER).states[:, -1]

    base = response(pulse, {})
    amplitude = float(max(abs(base - response(quiet, {}))))
    influence = []
    for i in range(spec["n_latent"]):
        key = f"z{i + 1}"
        pos = response(pulse, {key: 1e-3})
        neg = response(pulse, {key: -1e-3})
        influence.append(float(max(abs(pos - neg)) / 0.002))
    return {
        "input_effect_amplitude": amplitude,
        "max_output_derivative_by_coordinate": influence,
        "passed": 0.05 <= amplitude <= 8 and min(influence) >= 0.005,
    }


def scaling_cell(
    root: Path,
    release: Path,
    dimension: int,
    seed: int,
    variant: str,
    base: dict | None = None,
) -> dict:
    """Screen private systems and publish projected development channels."""
    key = "device_" + content_sha256([dimension, seed, variant])[:12]
    manifest = root / "evaluator" / key / "system.json"
    if manifest.exists():
        frozen = read_seal(manifest)
        spec, attempts = frozen["equations"], frozen["attempts"]
    else:
        attempts = []
        for index in range(8):
            spec = (
                sample_system(seed, index, dimension - 1)
                if base is None
                else dict(base)
            )
            if variant == "double_recurrent_coupling":
                spec["skew"] = (np.asarray(spec["skew"]) * 2).tolist()
            if variant == "wider_decay_spread":
                x = np.log(spec["decay"])
                spec["decay"] = np.exp(x.mean() + 2 * (x - x.mean())).tolist()
            try:
                witness = qualify(spec)
            except (ArithmeticError, ValueError, RuntimeError) as exc:
                witness = {"passed": False, "error": str(exc)}
            attempts.append(
                {"index": index, "equations": spec, "qualification": witness}
            )
            if witness["passed"] or base is not None:
                break
        seal(manifest, {"equations": spec, "attempts": attempts})
    row = {
        "cell": key,
        "family": "alien",
        "study": "scaling",
        "dimension": dimension,
        "generator_seed": seed,
        "variant": variant,
        "realization_sha256": content_sha256(spec),
        "qualification": attempts[-1]["qualification"],
        "attempt_count": len(attempts),
    }
    if not row["qualification"]["passed"]:
        return {**row, "status": "generation_failed"}
    splits = []
    for split, name in (("train", "train"), ("validation", "val")):
        path = root / "evaluator" / key / (split + ".json")
        if not path.exists():
            records = [
                device_row(_simulate_alien(p, spec, {}, SOLVER), i, name)
                for i, p in enumerate(device_protocols(split))
            ]
            seal(path, {"rows": records, "realization": row["realization_sha256"]})
        saved = read_seal(path)
        if saved["realization"] != row["realization_sha256"]:
            raise ValueError("procedural data belongs to a different draw")
        splits.append(saved["rows"])
    pub = read_seal(release / "public" / ALIEN / "specification.json")
    return {**row, **write_cell(root, key, pub, *splits)}
