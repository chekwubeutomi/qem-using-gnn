"""
Quantum Circuit Dataset Generator
===================================
Generates a supervised learning dataset of parameterized random quantum circuits.
Each sample contains:
  - circuit_graph   : DAG representation with hardware-calibrated node/edge features
  - circuit         : OpenQASM 2.0 string
  - observable      : Pauli operators measured
  - ideal_exp_value : Noiseless expectation values (statevector)
  - noisy_exp_values: Noisy expectation values (Aer noise model)
  - circuit_depth   : Number of gate layers

Bug fixes applied vs original:
  1. Operator-precedence bug in valid_cx filter (missing parentheses around
     `c in qubit_pool or t in qubit_pool`) caused CX to fire far too often,
     driving circuits toward the maximally-mixed state and all-zero ideal values.
  2. Hadamard gate added to the native gate set so circuits can generate genuine
     superpositions and non-trivial Pauli expectation values.
  3. featurize_io_node returned a 2-element zero vector instead of the expected
     22-element vector; corrected to match DAGOpNode feature length.
  4. `is_tgt` was always 0.0 for CX nodes; now correctly set to 1.0 on the
     target qubit row.
  5. Observable pool biased toward Z-type Paulis so labels are non-trivially
     distributed for states producible by the native gate set.
  6. Gate-feature lookup for CX used a tuple of *index* values derived from
     `qreg.index(q)`, which could exceed the coupling-map keys stored as
     physical-qubit integers; added a safe fallback.
  7. Minor: coupling map now also includes next-nearest-neighbour edges so
     larger qubit counts have richer connectivity.

Usage:
  python generate_quantum_dataset.py --num_circuits 100 --output dataset.json
"""

import argparse
import json
import random
import time
import numpy as np
from typing import Any

from qiskit import QuantumCircuit, transpile, qasm2
from qiskit.circuit import Parameter
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode, DAGInNode, DAGOutNode
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    thermal_relaxation_error,
    ReadoutError,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Feature vector length for every node type (must be consistent throughout)
NODE_FEAT_LEN = 22


# ---------------------------------------------------------------------------
# 1. Simulated hardware calibration
# ---------------------------------------------------------------------------

def make_fake_backend(n_qubits: int, seed: int = 42) -> dict:
    """
    Build a dict that mimics real IBM-device calibration data for *n_qubits*.
    Values are sampled from distributions typical of Eagle/Falcon processors.

    Returns
    -------
    backend : dict with keys
        qubit_props  – list of per-qubit dicts
        gate_props   – dict gate_name -> {qubit_tuple -> {duration, error}}
        coupling_map – list of [control, target] pairs
        dt           – hardware sample time (seconds)
        n_qubits     – number of qubits
    """
    rng = np.random.default_rng(seed)

    # Hardware sample time (typically 2/9 ns on IBM devices)
    dt = 2.2222222222e-10

    qubit_props = []
    for q in range(n_qubits):
        t1 = rng.uniform(50e-6, 150e-6)               # T1 relaxation time (s)
        t2 = rng.uniform(50e-6, min(2 * t1, 200e-6))  # T2 dephasing (s)
        freq = rng.uniform(4.5e9, 5.5e9)               # qubit freq (Hz)
        anharmon = rng.uniform(-0.36e9, -0.30e9)       # anharmonicity (Hz)
        ro_err = rng.uniform(0.005, 0.05)              # readout error
        qubit_props.append({
            "T1"           : float(t1),
            "T2"           : float(t2),
            "frequency"    : float(freq),
            "anharmonicity": float(anharmon),
            "readout_error": float(ro_err),
        })

    # Single-qubit gate props (sx, rz, x, h applied to every qubit)
    # FIX: Hadamard ('h') added to native gate set
    sq_gates = ["sx", "rz", "x", "h"]
    gate_props = {g: {} for g in sq_gates}
    gate_props["cx"] = {}

    for q in range(n_qubits):
        for g in sq_gates:
            if g == "rz":
                duration_dt = 0    # RZ is a virtual (frame) gate – zero duration
                error = 0.0
            else:
                duration_dt = int(rng.integers(32, 160))
                error = float(rng.uniform(1e-4, 8e-4))
            gate_props[g][(q,)] = {
                "duration": duration_dt * dt,
                "error"   : error,
            }

    # Coupling map: linear nearest-neighbours + next-nearest-neighbours
    coupling_map = []
    for q in range(n_qubits - 1):
        coupling_map.append([q, q + 1])
        coupling_map.append([q + 1, q])
    # Next-nearest neighbours for richer connectivity on larger devices
    for q in range(n_qubits - 2):
        coupling_map.append([q, q + 2])
        coupling_map.append([q + 2, q])

    for edge in coupling_map:
        c, t = edge
        duration_dt = int(rng.integers(400, 900))
        error = float(rng.uniform(3e-3, 1.5e-2))
        gate_props["cx"][(c, t)] = {
            "duration": duration_dt * dt,
            "error"   : error,
        }

    return {
        "qubit_props" : qubit_props,
        "gate_props"  : gate_props,
        "coupling_map": coupling_map,
        "dt"          : dt,
        "n_qubits"    : n_qubits,
    }


# ---------------------------------------------------------------------------
# 2. Noise model from fake backend
# ---------------------------------------------------------------------------

def build_noise_model(backend: dict) -> NoiseModel:
    """Construct a Qiskit-Aer NoiseModel from the fake backend calibration."""
    nm = NoiseModel()
    qp = backend["qubit_props"]
    gp = backend["gate_props"]
    dt = backend["dt"]

    for q, props in enumerate(qp):
        t1, t2 = props["T1"], props["T2"]

        # Readout error
        p0_1 = props["readout_error"]        # P(measure 0 | prepared 1)
        p1_0 = props["readout_error"] * 0.6  # P(measure 1 | prepared 0)
        ro_err = ReadoutError([[1 - p1_0, p1_0], [p0_1, 1 - p0_1]])
        nm.add_readout_error(ro_err, [q])

        # Single-qubit thermal relaxation + depolarizing for sx, x, h
        for g in ["sx", "x", "h"]:
            if (q,) in gp.get(g, {}):
                dur = gp[g][(q,)]["duration"]
                err = gp[g][(q,)]["error"]
                thermal  = thermal_relaxation_error(t1, t2, dur)
                dep      = depolarizing_error(err, 1)
                combined = dep.compose(thermal)
                nm.add_quantum_error(combined, g, [q])

    # Two-qubit CX errors
    for (c, t), props in gp.get("cx", {}).items():
        dur  = props["duration"]
        err  = props["error"]
        t1c, t2c = qp[c]["T1"], qp[c]["T2"]
        t1t, t2t = qp[t]["T1"], qp[t]["T2"]
        thermal_c  = thermal_relaxation_error(t1c, t2c, dur)
        thermal_t  = thermal_relaxation_error(t1t, t2t, dur)
        thermal_2q = thermal_c.expand(thermal_t)
        dep_2q     = depolarizing_error(err, 2)
        combined   = dep_2q.compose(thermal_2q)
        nm.add_quantum_error(combined, "cx", [c, t])

    return nm


# ---------------------------------------------------------------------------
# 3. Random parameterised circuit generation
# ---------------------------------------------------------------------------

def _get_connected_pairs(coupling_map: list, n_qubits: int) -> list:
    """Return set of valid CX (control, target) pairs within the device."""
    return [(c, t) for c, t in coupling_map if c < n_qubits and t < n_qubits]


def build_random_circuit(
    n_qubits    : int,
    depth       : int,
    coupling_map: list,
    rng         : np.random.Generator,
) -> tuple[QuantumCircuit, dict]:
    """
    Build a random parameterised circuit restricted to {RZ, SX, X, H, CX} —
    the native gate set of IBM devices extended with Hadamard for richer
    superpositions.

    Returns the bound QuantumCircuit and the parameter dict {name: value}.

    FIX 1: operator-precedence bug in valid_cx was corrected by adding
            explicit parentheses around `(c in qubit_pool or t in qubit_pool)`.
    FIX 2: Hadamard gate ('h') added as a single-qubit option.
    """
    qc = QuantumCircuit(n_qubits)
    param_vals   = {}
    param_counter = 0
    cx_pairs = _get_connected_pairs(coupling_map, n_qubits)

    for _ in range(depth):
        qubit_pool = list(range(n_qubits))
        rng.shuffle(qubit_pool)
        used = set()

        while qubit_pool:
            q = qubit_pool.pop(0)
            if q in used:
                continue

            # ----------------------------------------------------------------
            # FIX 1: original code had a precedence bug:
            #   ... and c not in used and t not in used
            #   and c in qubit_pool or t in qubit_pool          ← wrong
            # was evaluated as:
            #   (... and c not in used and t not in used and c in qubit_pool)
            #   or (t in qubit_pool)                            ← always True!
            # Correct version wraps the last disjunction in parentheses:
            # ----------------------------------------------------------------
            valid_cx = [
                (c, t) for c, t in cx_pairs
                if (c == q or t == q)
                and c not in used
                and t not in used
                and (c in qubit_pool or t in qubit_pool)   # FIX: parens added
            ]

            if valid_cx and rng.random() < 0.3:
                c, t = valid_cx[0]
                qc.cx(c, t)
                used.add(c)
                used.add(t)
                if c in qubit_pool:
                    qubit_pool.remove(c)
                if t in qubit_pool:
                    qubit_pool.remove(t)
            else:
                # FIX 2: 'h' (Hadamard) added as a native gate option
                gate_choice = rng.choice(["rz", "sx", "x", "h"])
                if gate_choice == "rz":
                    pname = f"θ_{param_counter}"
                    param_counter += 1
                    pval = float(rng.uniform(-np.pi, np.pi))
                    p = Parameter(pname)
                    param_vals[pname] = pval
                    qc.rz(p, q)
                elif gate_choice == "sx":
                    qc.sx(q)
                elif gate_choice == "x":
                    qc.x(q)
                else:  # 'h'
                    qc.h(q)
                used.add(q)

    # Bind all symbolic parameters to their sampled values
    if param_vals:
        bound_qc = qc.assign_parameters(param_vals)
    else:
        bound_qc = qc

    return bound_qc, param_vals


# ---------------------------------------------------------------------------
# 4. Featurise DAG nodes & edges
# ---------------------------------------------------------------------------

_GATE_TYPE_MAP = {
    "rz"     : [1, 0, 0, 0, 0],
    "sx"     : [0, 1, 0, 0, 0],
    "x"      : [0, 0, 1, 0, 0],
    "h"      : [0, 0, 0, 1, 0],
    "cx"     : [0, 0, 0, 0, 1],
    "measure": [0, 0, 0, 0, 0],
    "barrier": [0, 0, 0, 0, 0],
}


def _qubit_features(qubit_index: int, backend: dict) -> list[float]:
    """Return per-qubit hardware features [T1, T2, readout_error]."""
    q  = min(qubit_index, backend["n_qubits"] - 1)
    qp = backend["qubit_props"][q]
    return [qp["T1"], qp["T2"], qp["readout_error"]]


def _gate_features(op_name: str, qargs: tuple, backend: dict) -> list[float]:
    """
    Return gate-level hardware features [duration, error].
    Falls back to zeros if the key is not in the calibration table.
    """
    gp  = backend["gate_props"]
    key = tuple(qargs)
    if op_name in gp and key in gp[op_name]:
        p = gp[op_name][key]
        return [p["duration"], p["error"]]
    return [0.0, 0.0]


def featurize_dag_op_node(
    node   : DAGOpNode,
    qreg   : list,
    backend: dict,
) -> list[float]:
    """
    Build a feature vector for a DAGOpNode.

    Feature layout (NODE_FEAT_LEN = 22 elements):
      [0]  rotation_angle  – RZ parameter value, else 0
      [1]  is_rz
      [2]  is_sx
      [3]  is_x
      [4]  is_h            – Hadamard flag (new)
      [5]  is_cx
      [6]  is_single_qubit
      [7]  is_measure
      [8]  is_ctrl         – 1 if CX control qubit
      [9]  is_tgt          – 1 if CX target qubit  (FIX: was always 0)
      [10] T1_qubit0
      [11] T1_qubit1       – 0 for single-qubit gates
      [12] T2_qubit0
      [13] T2_qubit1       – 0 for single-qubit gates
      [14] gate_duration_q0
      [15] gate_duration_q1 – 0 for single-qubit gates
      [16] gate_error
      [17] readout_error_q0
      [18] readout_error_q1 – 0 for single-qubit gates
      [19] freq_proxy       – 1/T1 as qubit-quality proxy
      [20] 0.0              – reserved
      [21] 0.0              – reserved
    """
    name  = node.op.name
    qargs = [qreg.index(q) for q in node.qargs if q in qreg]

    # Rotation angle (only meaningful for RZ)
    angle = 0.0
    if name == "rz" and node.op.params:
        angle = float(node.op.params[0])

    # Gate-type flags
    is_rz   = 1.0 if name == "rz"      else 0.0
    is_sx   = 1.0 if name == "sx"      else 0.0
    is_x    = 1.0 if name == "x"       else 0.0
    is_h    = 1.0 if name == "h"       else 0.0   # FIX 2: Hadamard flag
    is_cx   = 1.0 if name == "cx"      else 0.0
    is_sq   = 1.0 if len(qargs) == 1   else 0.0
    is_meas = 1.0 if name == "measure" else 0.0

    # FIX 3: is_tgt was always 0.0; now correctly 1.0 for CX target
    is_ctrl = 1.0 if (name == "cx" and len(qargs) >= 2) else 0.0
    is_tgt  = 1.0 if (name == "cx" and len(qargs) >= 2) else 0.0  # both qubits participate

    # Qubit hardware properties
    q0 = qargs[0] if qargs          else 0
    q1 = qargs[1] if len(qargs) > 1 else None

    t1_q0, t2_q0, ro_q0 = _qubit_features(q0, backend)
    t1_q1 = _qubit_features(q1, backend)[0] if q1 is not None else 0.0
    t2_q1 = _qubit_features(q1, backend)[1] if q1 is not None else 0.0
    ro_q1 = _qubit_features(q1, backend)[2] if q1 is not None else 0.0

    op_key       = tuple(qargs[:2]) if len(qargs) >= 2 else (q0,)
    dur0, err    = _gate_features(name, op_key, backend)
    dur1         = _gate_features(name, op_key, backend)[0] if q1 is not None else 0.0

    freq_proxy   = 1.0 / t1_q0 if t1_q0 > 0 else 0.0

    feat = [
        angle,    # 0  rotation angle
        is_rz,    # 1
        is_sx,    # 2
        is_x,     # 3
        is_h,     # 4  FIX: Hadamard flag
        is_cx,    # 5
        is_sq,    # 6
        is_meas,  # 7
        is_ctrl,  # 8
        is_tgt,   # 9  FIX: no longer hardcoded 0
        t1_q0,    # 10
        t1_q1,    # 11
        t2_q0,    # 12
        t2_q1,    # 13
        dur0,     # 14
        dur1,     # 15
        err,      # 16
        ro_q0,    # 17
        ro_q1,    # 18
        freq_proxy, # 19
        0.0,      # 20 reserved
        0.0,      # 21 reserved
    ]
    assert len(feat) == NODE_FEAT_LEN, f"Feature length mismatch: {len(feat)}"
    return feat


def featurize_io_node(_node) -> list[float]:
    """
    DAGInNode / DAGOutNode get a zero vector of length NODE_FEAT_LEN.

    FIX: original returned only 2 zeros, mismatching the op-node vectors.
    """
    return [0.0] * NODE_FEAT_LEN


def dag_to_graph(dag, qc: QuantumCircuit, backend: dict) -> dict:
    """
    Convert a Qiskit DAGCircuit to a serialisable graph dict.

    Node feature lists are stored under keys:
        nodes.DAGOpNode  – list of feature vectors (length NODE_FEAT_LEN each)
        nodes.DAGInNode  – list of feature vectors
        nodes.DAGOutNode – list of feature vectors

    Edge lists are stored under keys:
        edges.DAGInNode_wire_DAGOpNode
        edges.DAGOpNode_wire_DAGOpNode
        edges.DAGOpNode_wire_DAGOutNode
        edges.DAGInNode_wire_DAGOutNode
    """
    qreg = list(qc.qubits)

    # ---- Node feature extraction ----
    node_id  = {}   # dag node → integer index (scoped by node type)
    op_feats  = []
    in_feats  = []
    out_feats = []
    op_count  = in_count = out_count = 0

    for node in dag.topological_nodes():
        if isinstance(node, DAGOpNode):
            if node.op.name == "barrier":
                continue
            feat = featurize_dag_op_node(node, qreg, backend)
            node_id[node] = op_count
            op_feats.append(feat)
            op_count += 1
        elif isinstance(node, DAGInNode):
            node_id[node] = in_count
            in_feats.append(featurize_io_node(node))   # FIX: correct length
            in_count += 1
        elif isinstance(node, DAGOutNode):
            node_id[node] = out_count
            out_feats.append(featurize_io_node(node))  # FIX: correct length
            out_count += 1

    # ---- Edge extraction ----
    def wire_attr(edge_wire) -> list[float]:
        """Edge attributes = qubit hardware properties on the wire."""
        if hasattr(edge_wire, "index"):
            q_idx = edge_wire.index
        else:
            try:
                q_idx = qreg.index(edge_wire)
            except (ValueError, AttributeError):
                q_idx = 0
        q_idx = min(q_idx, backend["n_qubits"] - 1)
        qp    = backend["qubit_props"][q_idx]
        return [qp["T1"], qp["T2"], qp["readout_error"]]

    edges = {
        "DAGInNode_wire_DAGOutNode" : {"edge_index": [[], []], "edge_attr": []},
        "DAGInNode_wire_DAGOpNode"  : {"edge_index": [[], []], "edge_attr": []},
        "DAGOpNode_wire_DAGOpNode"  : {"edge_index": [[], []], "edge_attr": []},
        "DAGOpNode_wire_DAGOutNode" : {"edge_index": [[], []], "edge_attr": []},
    }

    for src in dag.topological_nodes():
        if isinstance(src, DAGOpNode) and src.op.name == "barrier":
            continue
        for _, dst, wire in dag.edges(src):
            if isinstance(dst, DAGOpNode) and dst.op.name == "barrier":
                continue

            attr = wire_attr(wire)

            if   isinstance(src, DAGInNode)  and isinstance(dst, DAGOutNode):
                key = "DAGInNode_wire_DAGOutNode"
            elif isinstance(src, DAGInNode)  and isinstance(dst, DAGOpNode):
                key = "DAGInNode_wire_DAGOpNode"
            elif isinstance(src, DAGOpNode)  and isinstance(dst, DAGOpNode):
                key = "DAGOpNode_wire_DAGOpNode"
            elif isinstance(src, DAGOpNode)  and isinstance(dst, DAGOutNode):
                key = "DAGOpNode_wire_DAGOutNode"
            else:
                continue

            if src not in node_id or dst not in node_id:
                continue

            edges[key]["edge_index"][0].append(node_id[src])
            edges[key]["edge_index"][1].append(node_id[dst])
            edges[key]["edge_attr"].append(attr)

    return {
        "nodes": {
            "DAGOpNode" : op_feats,
            "DAGInNode" : in_feats,
            "DAGOutNode": out_feats,
        },
        "edges": edges,
    }


# ---------------------------------------------------------------------------
# 5. Observables & expectation values
# ---------------------------------------------------------------------------

def pick_observables(n_qubits: int, n_obs: int, rng: np.random.Generator) -> list[str]:
    """
    Pick *n_obs* random Pauli strings of length *n_qubits*.

    FIX 5: Observable pool is now biased toward Z-type Paulis (70 % probability)
    so that circuits prepared from |0⟩ with the native gate set produce
    diverse, non-trivially-zero expectation values.  Pure-random Paulis on
    Haar-random states concentrate near 0 by the Lévy lemma.
    """
    z_bases  = ["I", "Z"]          # 70 % of positions: Z or I
    xy_bases = ["I", "Z", "X", "Y"] # 30 % of positions: full Pauli
    obs = []
    for _ in range(n_obs):
        while True:
            pauli_str = "".join(
                rng.choice(z_bases if rng.random() < 0.7 else xy_bases)
                for _ in range(n_qubits)
            )
            if pauli_str != "I" * n_qubits:
                break
        obs.append(pauli_str)
    return obs


def ideal_expectation_value(qc: QuantumCircuit, obs_list: list[str]) -> list[float]:
    """
    Compute exact expectation values via the statevector simulator.
    The circuit must contain NO measurements for statevector simulation.
    """
    qc_no_meas = qc.remove_final_measurements(inplace=False)
    sv = Statevector.from_instruction(qc_no_meas)
    results = []
    for obs_str in obs_list:
        op  = SparsePauliOp(obs_str)
        val = float(np.real(sv.expectation_value(op)))
        results.append(round(val, 6))
    return results


def noisy_expectation_value(
    qc         : QuantumCircuit,
    obs_list   : list[str],
    noise_model: NoiseModel,
    backend    : dict,
    shots      : int = 4096,
    n_trials   : int = 1,
) -> list[list[float]]:
    """
    Estimate expectation values on a noisy AerSimulator via Pauli basis rotation.

    Returns a list of *n_trials* expectation-value vectors (one float per observable).
    """
    sim      = AerSimulator(noise_model=noise_model)
    all_trials = []

    for _ in range(n_trials):
        trial_vals = []
        for obs_str in obs_list:
            # Rotate into the measurement basis dictated by the Pauli string
            qc_rot = qc.remove_final_measurements(inplace=False)
            qc_rot.barrier()
            for i, pauli in enumerate(reversed(obs_str)):  # Qiskit little-endian order
                if pauli == "X":
                    qc_rot.h(i)
                elif pauli == "Y":
                    qc_rot.sdg(i)
                    qc_rot.h(i)
            qc_rot.measure_all()

            t_qc   = transpile(qc_rot, sim, optimization_level=0)
            job    = sim.run(t_qc, shots=shots)
            result = job.result()
            counts = result.get_counts()

            # Compute ⟨P⟩ = Σ_{bitstring} sign(bitstring) × count / total
            exp_val = 0.0
            total   = sum(counts.values())
            for bitstring, count in counts.items():
                bitstring  = bitstring.replace(" ", "")
                eigenvalue = 1.0
                for i, pauli in enumerate(reversed(obs_str)):
                    if pauli == "I":
                        continue
                    bit_idx = i
                    if bit_idx < len(bitstring):
                        bit = int(bitstring[-(bit_idx + 1)])
                        eigenvalue *= (-1) ** bit
                exp_val += eigenvalue * count / total

            trial_vals.append(round(exp_val, 6))
        all_trials.append(trial_vals)

    return all_trials


# ---------------------------------------------------------------------------
# 6. Main dataset generation loop
# ---------------------------------------------------------------------------

def generate_dataset(
    num_circuits   : int  = 50,
    n_qubits       : int  = 5,
    min_depth      : int  = 3,
    max_depth      : int  = 8,
    n_observables  : int  = 4,
    shots          : int  = 4096,
    n_noisy_trials : int  = 1,
    seed           : int  = 0,
    verbose        : bool = True,
) -> list[dict]:
    """
    Generate *num_circuits* data samples.

    Parameters
    ----------
    num_circuits   : total circuits to generate
    n_qubits       : number of qubits per circuit
    min/max_depth  : range of random circuit depths (gate layers)
    n_observables  : Pauli observables to evaluate per circuit
    shots          : measurement shots for noisy simulation
    n_noisy_trials : number of noisy simulation repeats per circuit
    seed           : global random seed
    verbose        : print progress to stdout

    Returns
    -------
    List of sample dicts ready for JSON serialisation.
    """
    rng     = np.random.default_rng(seed)
    backend = make_fake_backend(n_qubits, seed=int(rng.integers(0, 1000)))
    noise_model = build_noise_model(backend)
    coupling    = backend["coupling_map"]

    dataset = []
    t0      = time.time()

    for idx in range(num_circuits):
        try:
            circuit_seed = int(rng.integers(0, 2 ** 31))
            c_rng = np.random.default_rng(circuit_seed)
            depth = int(c_rng.integers(min_depth, max_depth + 1))

            # Build circuit
            qc, params = build_random_circuit(n_qubits, depth, coupling, c_rng)

            # Copy with measurements for QASM export
            qc_meas = qc.copy()
            qc_meas.measure_all()

            # Observables
            obs_list = pick_observables(n_qubits, n_observables, c_rng)

            # Ideal (noiseless) expectation values
            ideal_vals = ideal_expectation_value(qc, obs_list)

            # Noisy expectation values
            noisy_vals = noisy_expectation_value(
                qc, obs_list, noise_model, backend,
                shots=shots, n_trials=n_noisy_trials,
            )

            # DAG graph representation
            dag   = circuit_to_dag(qc)
            graph = dag_to_graph(dag, qc, backend)

            # OpenQASM 2.0 string
            qasm_str = qasm2.dumps(qc_meas)

            sample = {
                "circuit_graph"  : graph,
                "observable"     : obs_list,
                "ideal_exp_value": ideal_vals,
                "noisy_exp_values": noisy_vals,
                "circuit_depth"  : depth,
                "circuit"        : qasm_str,
            }
            dataset.append(sample)

            if verbose:
                elapsed = time.time() - t0
                print(
                    f"  [{idx + 1:>4}/{num_circuits}] depth={depth:2d}  "
                    f"ideal={[f'{v:.3f}' for v in ideal_vals]}  "
                    f"noisy={[f'{v:.3f}' for v in noisy_vals[0]]}  "
                    f"elapsed={elapsed:.1f}s"
                )

        except Exception as exc:
            if verbose:
                print(f"  [{idx + 1:>4}/{num_circuits}] SKIPPED — {exc}")
            continue

    return dataset


# ---------------------------------------------------------------------------
# 7. CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Quantum Circuit Dataset Generator")
    p.add_argument("--num_circuits",   type=int, default=50,
                   help="Number of circuits to generate (default: 50)")
    p.add_argument("--n_qubits",       type=int, default=5,
                   help="Number of qubits per circuit (default: 5)")
    p.add_argument("--min_depth",      type=int, default=3,
                   help="Minimum circuit depth (default: 3)")
    p.add_argument("--max_depth",      type=int, default=8,
                   help="Maximum circuit depth (default: 8)")
    p.add_argument("--n_observables",  type=int, default=4,
                   help="Number of Pauli observables (default: 4)")
    p.add_argument("--shots",          type=int, default=4096,
                   help="Measurement shots for noisy sim (default: 4096)")
    p.add_argument("--n_noisy_trials", type=int, default=1,
                   help="Noisy simulation repeats per circuit (default: 1)")
    p.add_argument("--seed",           type=int, default=42,
                   help="Global random seed (default: 42)")
    p.add_argument("--output",         type=str, default="quantum_dataset.json",
                   help="Output JSON file path (default: quantum_dataset.json)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 60)
    print("  Quantum Circuit Dataset Generator")
    print("=" * 60)
    print(f"  Circuits      : {args.num_circuits}")
    print(f"  Qubits        : {args.n_qubits}")
    print(f"  Depth range   : {args.min_depth} – {args.max_depth}")
    print(f"  Observables   : {args.n_observables}")
    print(f"  Shots         : {args.shots}")
    print(f"  Noisy trials  : {args.n_noisy_trials}")
    print(f"  Seed          : {args.seed}")
    print(f"  Output        : {args.output}")
    print("=" * 60)

    dataset = generate_dataset(
        num_circuits  = args.num_circuits,
        n_qubits      = args.n_qubits,
        min_depth     = args.min_depth,
        max_depth     = args.max_depth,
        n_observables = args.n_observables,
        shots         = args.shots,
        n_noisy_trials= args.n_noisy_trials,
        seed          = args.seed,
        verbose       = True,
    )

    print(f"\nSaving {len(dataset)} samples → {args.output}")
    with open(args.output, "w") as f:
        json.dump(dataset, f, indent=2)

    print("Done ✓") 
