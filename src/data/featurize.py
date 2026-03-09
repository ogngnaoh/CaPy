"""SMILES → PyG molecular graphs using OGB-style integer features.

WHY THIS WORKS
--------------
We store atom/bond features as *integer indices* rather than one-hot
vectors.  This is the convention used by Open Graph Benchmark (OGB) and
lets the downstream GIN encoder use ``torch.nn.Embedding`` tables —
which are more memory-efficient and faster than linear layers on
sparse one-hot inputs.

Each atom gets a 9-element integer vector; each bond gets a 4-element
integer vector.  We include *both* directions for every bond
(i → j AND j → i) because message-passing GNNs need undirected edges
represented as two directed edges.

The ``ATOM_FEATURE_DIMS`` and ``BOND_FEATURE_DIMS`` lists export the
vocabulary size per feature slot, which the encoder needs to size its
embedding tables.
"""

from __future__ import annotations

from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Lazy heavy imports — only resolved when functions are actually called.
# ---------------------------------------------------------------------------

_torch = None
_Chem = None
_Data = None


def _ensure_imports() -> None:
    """Import torch, rdkit, and torch_geometric on first use."""
    global _torch, _Chem, _Data  # noqa: PLW0603
    if _torch is None:
        import torch

        _torch = torch
    if _Chem is None:
        from rdkit import Chem

        _Chem = Chem
    if _Data is None:
        from torch_geometric.data import Data

        _Data = Data


# ---------------------------------------------------------------------------
# Feature vocabularies (integer index → category)
# ---------------------------------------------------------------------------

# Canonical atomic numbers we care about; everything else → OTHER (index 9)
_ATOMIC_NUM_LIST = [6, 7, 8, 9, 15, 16, 17, 35, 53]  # C N O F P S Cl Br I
_ATOMIC_NUM_TO_IDX = {a: i for i, a in enumerate(_ATOMIC_NUM_LIST)}
_ATOMIC_NUM_OTHER = len(_ATOMIC_NUM_LIST)  # 9

# Chirality types
_CHIRALITY_LIST = ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW"]
_CHIRALITY_OTHER = len(_CHIRALITY_LIST)  # 3

# Hybridization types
_HYBRIDIZATION_LIST = ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2"]
_HYBRIDIZATION_TO_IDX = {h: i for i, h in enumerate(_HYBRIDIZATION_LIST)}
_HYBRIDIZATION_OTHER = len(_HYBRIDIZATION_LIST)  # 6

# Bond types
_BOND_TYPE_LIST = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
_BOND_TYPE_TO_IDX = {b: i for i, b in enumerate(_BOND_TYPE_LIST)}

# Bond stereo types
_STEREO_LIST = [
    "STEREONONE",
    "STEREOANY",
    "STEREOZ",
    "STEREOE",
    "STEREOCIS",
    "STEREOTRANS",
]
_STEREO_TO_IDX = {s: i for i, s in enumerate(_STEREO_LIST)}

# ---------------------------------------------------------------------------
# Public constants — consumed by the encoder to build Embedding tables
# ---------------------------------------------------------------------------

#: Vocabulary size per atom feature slot (9 features).
ATOM_FEATURE_DIMS = [
    len(_ATOMIC_NUM_LIST) + 1,  # atomic number (10)
    len(_CHIRALITY_LIST) + 1,  # chirality (4)
    6,  # degree 0–5
    5,  # formal charge −2..+2 → 0..4
    5,  # num H 0–4
    3,  # num radical electrons 0–2
    len(_HYBRIDIZATION_LIST) + 1,  # hybridization (7, last = OTHER)
    2,  # is aromatic
    2,  # is in ring
]

#: Vocabulary size per bond feature slot (4 features).
BOND_FEATURE_DIMS = [
    len(_BOND_TYPE_LIST),  # bond type (4)
    2,  # is conjugated
    2,  # is in ring
    len(_STEREO_LIST),  # stereo (6)
]

NUM_ATOM_FEATURES: int = len(ATOM_FEATURE_DIMS)  # 9
NUM_BOND_FEATURES: int = len(BOND_FEATURE_DIMS)  # 4


# ---------------------------------------------------------------------------
# Per-atom / per-bond feature extraction
# ---------------------------------------------------------------------------


def _atom_to_feature_vector(atom) -> list[int]:  # noqa: ANN001
    """Convert an RDKit ``Atom`` to a 9-element integer feature vector.

    Args:
        atom: An ``rdkit.Chem.Atom`` object.

    Returns:
        List of 9 integer indices.
    """
    return [
        _ATOMIC_NUM_TO_IDX.get(atom.GetAtomicNum(), _ATOMIC_NUM_OTHER),
        (
            _CHIRALITY_LIST.index(str(atom.GetChiralTag()))
            if str(atom.GetChiralTag()) in _CHIRALITY_LIST
            else _CHIRALITY_OTHER
        ),
        min(atom.GetTotalDegree(), 5),
        min(max(atom.GetFormalCharge() + 2, 0), 4),  # shift -2..+2 → 0..4, clamped
        min(atom.GetTotalNumHs(), 4),
        min(atom.GetNumRadicalElectrons(), 2),
        _HYBRIDIZATION_TO_IDX.get(str(atom.GetHybridization()), _HYBRIDIZATION_OTHER),
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
    ]


def _bond_to_feature_vector(bond) -> list[int]:  # noqa: ANN001
    """Convert an RDKit ``Bond`` to a 4-element integer feature vector.

    Args:
        bond: An ``rdkit.Chem.Bond`` object.

    Returns:
        List of 4 integer indices.
    """
    return [
        _BOND_TYPE_TO_IDX.get(str(bond.GetBondType()), 0),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
        _STEREO_TO_IDX.get(str(bond.GetStereo()), 0),
    ]


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def featurize_atoms(mol) -> torch.Tensor:  # noqa: ANN001, F821
    """Build the atom feature matrix for a molecule.

    Args:
        mol: An RDKit ``Mol`` object.

    Returns:
        Long tensor of shape ``[num_atoms, 9]``.
    """
    _ensure_imports()
    rows = [_atom_to_feature_vector(atom) for atom in mol.GetAtoms()]
    return _torch.tensor(rows, dtype=_torch.long)


def featurize_bonds(
    mol,  # noqa: ANN001
) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: F821
    """Build edge_index and edge_attr for a molecule.

    Both directions are included for each bond (i→j and j→i) so that
    message-passing GNNs treat the graph as undirected.

    Args:
        mol: An RDKit ``Mol`` object.

    Returns:
        Tuple of (edge_index ``[2, num_edges]``, edge_attr ``[num_edges, 4]``).
        For molecules with no bonds, returns tensors with 0 edges.
    """
    _ensure_imports()
    if mol.GetNumBonds() == 0:
        edge_index = _torch.zeros((2, 0), dtype=_torch.long)
        edge_attr = _torch.zeros((0, NUM_BOND_FEATURES), dtype=_torch.long)
        return edge_index, edge_attr

    src, dst, attrs = [], [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        feat = _bond_to_feature_vector(bond)
        # Both directions
        src += [i, j]
        dst += [j, i]
        attrs += [feat, feat]

    edge_index = _torch.tensor([src, dst], dtype=_torch.long)
    edge_attr = _torch.tensor(attrs, dtype=_torch.long)
    return edge_index, edge_attr


def smiles_to_graph(smiles: str) -> Data | None:  # noqa: F821
    """Convert a SMILES string to a PyG ``Data`` object.

    Args:
        smiles: A SMILES string representing a molecule.

    Returns:
        A ``torch_geometric.data.Data`` with ``x``, ``edge_index``,
        ``edge_attr``, and ``smiles`` fields, or ``None`` if the SMILES
        cannot be parsed.
    """
    _ensure_imports()
    if not smiles or not smiles.strip():
        logger.warning("Could not parse SMILES: %s", smiles)
        return None
    mol = _Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.warning("Could not parse SMILES: %s", smiles)
        return None

    x = featurize_atoms(mol)
    edge_index, edge_attr = featurize_bonds(mol)

    return _Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        smiles=smiles,
    )


def featurize_dataset(
    smiles_list: list[str],
    compound_ids: list[str],
) -> dict[str, Data]:  # noqa: F821
    """Featurize a list of SMILES into a compound_id → Data mapping.

    Args:
        smiles_list: SMILES strings, one per compound.
        compound_ids: Unique identifiers aligned with *smiles_list*.

    Returns:
        Dict mapping compound_id to PyG ``Data`` objects.
        Compounds whose SMILES fail to parse are omitted.
    """
    _ensure_imports()
    graphs: dict[str, Data] = {}  # noqa: F821
    failed = 0
    for cid, smi in zip(compound_ids, smiles_list):
        g = smiles_to_graph(smi)
        if g is not None:
            g.compound_id = cid
            graphs[cid] = g
        else:
            failed += 1

    logger.info(
        "Featurized %d / %d compounds (%d failed).",
        len(graphs),
        len(smiles_list),
        failed,
    )
    if len(graphs) == 0 and len(smiles_list) > 0:
        raise ValueError(
            f"All {len(smiles_list)} SMILES failed to featurize. "
            "Check that SMILES are valid and rdkit is installed correctly."
        )
    return graphs
