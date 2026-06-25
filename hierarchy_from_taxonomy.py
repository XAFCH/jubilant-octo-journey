# hierarchy_from_taxonomy.py

from __future__ import annotations

from typing import Dict, List, Tuple, Set
from collections import defaultdict

import torch

Path = Tuple[str, ...]   # e.g. ("Fiction", "Western Fiction")


# ----------------------------------------------------------------------
# Basic taxonomy parsing
# ----------------------------------------------------------------------

def load_taxonomy(
    taxonomy_path: str,
) -> Dict[str, List[str]]:
    """
    Load the taxonomy file and build a parent -> [children] adjacency map.

    Expected format (TSV):
        Parent<TAB>Child1<TAB>Child2<...>

    Example lines:
        Root    Children’s Books    Poetry   Fiction   Nonfiction   ...
        Fiction Fantasy Spiritual Fiction Literary Fiction ...

    Returns:
        parent_children: dict mapping parent label -> list of child labels.
    """
    parent_children: Dict[str, List[str]] = defaultdict(list)

    with open(taxonomy_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            parent = parts[0].strip()
            children = [p.strip() for p in parts[1:] if p.strip()]
            if not children:
                continue
            parent_children[parent].extend(children)

    return parent_children


def find_roots(parent_children: Dict[str, List[str]]) -> List[str]:
    """
    Find root nodes (those that never appear as children).
    In your taxonomy this should just be ["Root"].
    """
    all_parents = set(parent_children.keys())
    all_children = {c for children in parent_children.values() for c in children}
    roots = [p for p in all_parents if p not in all_children]
    return roots


# ----------------------------------------------------------------------
# Label set loading (labels.txt)
# ----------------------------------------------------------------------

def load_label_names_flat(labels_path: str) -> Set[str]:
    """
    Load all label names from labels.txt.

    Expected format:
        One label per line (no tabs), e.g.:
            Urban Fantasy
            Parenting
            World History
            ...
            Fiction
            Nonfiction
            Root

    Returns:
        Set of label strings.
    """
    labels: Set[str] = set()
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if not name:
                continue
            labels.add(name)
    return labels


# ----------------------------------------------------------------------
# Path enumeration
# ----------------------------------------------------------------------

def enumerate_all_node_paths(
    parent_children: Dict[str, List[str]],
    root_name: str = "Root",
    drop_root: bool = True,
) -> Dict[str, Path]:
    """
    DFS over the taxonomy to compute the path from the root to *every* node.

    Args:
        parent_children: parent -> [children] map.
        root_name:       name of the artificial root in the taxonomy.
        drop_root:       if True, returned paths do not include 'Root' itself.

    Returns:
        node_to_path: dict mapping node label -> Path (tuple of labels)
                      representing the path from root to that node.
                      (Root itself maps to () if drop_root=True.)
    """
    node_to_path: Dict[str, Path] = {}

    def dfs(node: str, current_path: List[str]):
        # current_path is WITHOUT 'Root' if drop_root=True
        node_to_path[node] = tuple(current_path)
        children = parent_children.get(node, [])
        for c in children:
            dfs(c, current_path + [c])

    if drop_root:
        dfs(root_name, [])
    else:
        dfs(root_name, [root_name])

    return node_to_path


def enumerate_label_to_paths(
    parent_children: Dict[str, List[str]],
    root_name: str = "Root",
    drop_root: bool = True,
) -> Dict[str, List[Path]]:
    """Enumerate *all* paths for each label, even if the same label appears under multiple parents.

    Returns:
        label_to_paths: dict mapping label string -> list of Path tuples for every occurrence.

    Note:
        This is necessary for taxonomies where a label name (e.g., 'Europe') appears
        in multiple places. A simple node->path dict would overwrite duplicates.
    """
    label_to_paths: Dict[str, List[Path]] = defaultdict(list)

    start_path: List[str] = [] if drop_root else [root_name]
    stack: List[Tuple[str, List[str], Set[str]]] = [(root_name, start_path, {root_name})]

    while stack:
        node, path_so_far, on_path = stack.pop()

        # Store the current path for this node label.
        label_to_paths[node].append(tuple(path_so_far))

        for child in parent_children.get(node, []):
            if child in on_path:
                cycle_preview = " -> ".join(list(path_so_far) + [child])
                raise ValueError(
                    "Cycle detected in taxonomy while enumerating paths. "
                    f"Example cycle segment: {cycle_preview}"
                )

            child_path = path_so_far + [child]
            child_on_path = set(on_path)
            child_on_path.add(child)
            stack.append((child, child_path, child_on_path))

    return label_to_paths


def common_prefix_length(p1: Path, p2: Path) -> int:
    """Length of the common prefix between two paths."""
    n = min(len(p1), len(p2))
    for i in range(n):
        if p1[i] != p2[i]:
            return i
    return n

def compute_distance_matrix_from_paths(paths: List[Path]) -> torch.Tensor:
    """
    Compute integer path distance matrix:
        d(i, j) = len(p_i) + len(p_j) - 2 * common_prefix_length(p_i, p_j)

    Returns:
        dist_matrix: [P, P] torch.long tensor
    """
    P = len(paths)
    dist = torch.zeros(P, P, dtype=torch.long)

    for i in range(P):
        pi = paths[i]
        di = len(pi)
        for j in range(i, P):
            pj = paths[j]
            dj = len(pj)
            lca = common_prefix_length(pi, pj)
            d_ij = di + dj - 2 * lca
            dist[i, j] = d_ij
            dist[j, i] = d_ij

    return dist


def build_distance_level_index(
    dist_matrix: torch.Tensor,
) -> List[Dict[int, List[int]]]:
    """
    Build per-path distance-level index.

    Returns:
        level_index: list of length P
            level_index[i] is a dict:
                distance_value -> list[path_id]
            Example:
                level_index[3] = {
                    1: [5, 7, 9],
                    2: [1, 2, 8],
                    3: [0, 4],
                }
    """
    if dist_matrix.dim() != 2 or dist_matrix.size(0) != dist_matrix.size(1):
        raise ValueError("dist_matrix must be a square [P, P] tensor.")

    P = dist_matrix.size(0)
    level_index: List[Dict[int, List[int]]] = []

    for i in range(P):
        by_level: Dict[int, List[int]] = defaultdict(list)
        for j in range(P):
            if i == j:
                continue
            d = int(dist_matrix[i, j].item())
            by_level[d].append(j)

        level_index.append(
            {d: ids for d, ids in sorted(by_level.items(), key=lambda x: x[0])}
        )

    return level_index

# def compute_w_matrix_from_paths(
#     paths: List[Path],
#     lam: float = 0.5,
# ) -> torch.Tensor:
#     """
#     Compute global w_matrix over label paths (including internal nodes).
#
#     Distance:
#         d(i, j) = len(p_i) + len(p_j) - 2 * common_prefix_length(p_i, p_j)
#     Weight:
#         w(i, j) = exp(-λ * d(i, j))
#
#     Args:
#         paths: list of paths (tuples of labels).
#         lam:   lambda in exp(-λ d).
#
#     Returns:
#         w_matrix: [P, P] tensor where P = len(paths).
#     """
#     P = len(paths)
#     dist = torch.zeros(P, P, dtype=torch.float32)
#
#     for i in range(P):
#         pi = paths[i]
#         di = len(pi)
#         for j in range(P):
#             pj = paths[j]
#             dj = len(pj)
#             lca = common_prefix_length(pi, pj)
#             d_ij = di + dj - 2 * lca
#             dist[i, j] = float(d_ij)
#
#     w_matrix = torch.exp(-lam * dist)
#     return w_matrix
def compute_w_matrix_from_paths(
    paths: List[Path],
    lam: float = 0.5,
) -> torch.Tensor:
    """
    Compute global w_matrix over label paths (including internal nodes).

    Distance:
        d(i, j) = len(p_i) + len(p_j) - 2 * common_prefix_length(p_i, p_j)
    Weight:
        w(i, j) = exp(-λ * d(i, j))

    Args:
        paths: list of paths (tuples of labels).
        lam:   lambda in exp(-λ d).

    Returns:
        w_matrix: [P, P] tensor where P = len(paths).
    """
    dist = compute_distance_matrix_from_paths(paths).to(torch.float32)
    w_matrix = torch.exp(-lam * dist)
    return w_matrix

# ----------------------------------------------------------------------
# High-level helper
# ----------------------------------------------------------------------

def prepare_hierarchy_from_taxonomy(
    taxonomy_path: str,
    labels_path: str | None = None,
    root_name: str = "Root",
    lam: float = 0.5,
):
    """
    Build global hierarchical paths and w_matrix from taxonomy, including
    internal nodes that are actually used as labels.

    Args:
        taxonomy_path: path to taxonomy file (e.g. bgc.taxonomy).
        labels_path:   path to labels.txt listing all label names that appear
                       as targets in the dataset(s). If provided, we only
                       keep paths for these nodes (excluding Root).
                       If None, we fall back to root->leaf paths only.
        root_name:     name of artificial root in taxonomy.
        lam:           lambda in exp(-λ d) for w_matrix.

    Returns:
        paths:      list[Path] of all label paths used as targets, e.g.:
                    ("Fiction","Western Fiction"),
                    ("Children's Books",),
                    ("Nonfiction","History","World History"), ...
        path_to_id: dict[Path, int] mapping each path -> unique path_id.
        w_matrix:   [P, P] tensor of hierarchy-based similarities.
    """
    # 1) Parse taxonomy
    parent_children = load_taxonomy(taxonomy_path)
    roots = find_roots(parent_children)
    if root_name not in roots:
        raise ValueError(f"Expected root '{root_name}' in taxonomy, got roots={roots}")

    # 2) Compute paths from root to every node occurrence (supports repeated labels)
    label_to_paths = enumerate_label_to_paths(
        parent_children,
        root_name=root_name,
        drop_root=True,   # we don't want "Root" in the paths
    )

    # 3) Determine which labels we actually care about
    if labels_path is not None:
        label_names = load_label_names_flat(labels_path)
        target_labels = [n for n in label_names if n != root_name]
    else:
        # fallback: use leaves only (labels that have no children)
        # NOTE: this is label-level and may include internal nodes if taxonomy is inconsistent.
        all_parents = set(parent_children.keys())
        all_children = {c for children in parent_children.values() for c in children}
        target_labels = [n for n in all_children if n not in all_parents]

    # 4) Build paths list for all occurrences of the target labels
    paths: List[Path] = []
    seen: Set[Path] = set()
    for label in sorted(target_labels):
        for path in label_to_paths.get(label, []):
            if len(path) == 0:
                # exclude root
                continue
            if path not in seen:
                seen.add(path)
                paths.append(path)

    if not paths:
        raise ValueError("No target paths were found. Check labels_path and taxonomy.")

    # 5) Assign IDs and compute w_matrix
    path_to_id: Dict[Path, int] = {p: i for i, p in enumerate(paths)}
    w_matrix = compute_w_matrix_from_paths(paths, lam=lam)

    return paths, path_to_id, w_matrix