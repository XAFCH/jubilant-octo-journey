from typing import Set


def load_label_names(label_txt_path: str) -> Set[str]:
    """
    Load all label names from a TSV-like label file such as hierarchy.txt.

    Format:
      - Lines can be "Parent<TAB>Child" (most lines)
      - Or singletons like "Humor", "Poetry" (no tab).

    Returns:
        A set of unique label names (both parents and children).
    """
    labels = set()

    with open(label_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            for p in parts:
                p = p.strip()
                if p:
                    labels.add(p)

    return labels