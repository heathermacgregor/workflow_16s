# ===================================== IMPORTS ====================================== #

# Standard Library Imports
from __future__ import annotations

import logging
import os
import re
import warnings
import zipfile
from pathlib import Path
from typing import Dict, Iterator, List, Pattern, Tuple, Union
from urllib.parse import urljoin

# Third-Party Imports
import requests
from bs4 import BeautifulSoup
from rich.progress import Progress, TaskID

# ================================== LOCAL IMPORTS =================================== #
# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger('workflow_16s')
warnings.filterwarnings("ignore")  # Suppress warnings

# ==================================== FUNCTIONS ===================================== #

def _scrape_latest_zip_url(base_url: str) -> Tuple[str, str]:
    """
    Return *(zip_url, version_str)* for the newest *FAPROTAX_*.zip on the page.
    """
    soup = BeautifulSoup(requests.get(base_url, timeout=30).text, "html.parser")

    # collect all links of the form ".../FAPROTAX_<ver>.zip"
    zip_links: list[Tuple[str, Tuple[int, ...]]] = []
    regex = re.compile(r"FAPROTAX_(\d+(?:\.\d+)*)\.zip$", re.I)

    for a in soup.find_all("a", href=True):
        m = regex.search(a["href"])
        if m:
            ver_tuple = tuple(int(p) for p in m.group(1).split("."))
            zip_links.append((urljoin(base_url, a["href"]), ver_tuple))

    if not zip_links:  # pragma: no cover
        raise RuntimeError("No FAPROTAX *.zip links found on the download page.")

    # pick the link with the highest version tuple
    zip_links.sort(key=lambda x: x[1], reverse=True)
    latest_url, latest_ver = zip_links[0]
    return latest_url, ".".join(map(str, latest_ver))


def _extract_faprotax_txt(zip_path: Path, out_folder: Path) -> Path:
    """
    Extract *FAPROTAX.txt* from *zip_path* into *out_folder*.
    Returns the path of the extracted file.
    """
    with zipfile.ZipFile(zip_path) as zf:
        txt_members = [m for m in zf.namelist() if m.endswith("FAPROTAX.txt")]
        if not txt_members:  # pragma: no cover
            raise RuntimeError("FAPROTAX.txt not found inside ZIP archive.")
        member = txt_members[0]
        extracted_path = zf.extract(member, path=out_folder)
        dst = out_folder / "FAPROTAX.txt"
        Path(extracted_path).replace(dst)  # move/overwrite for consistent name
        return dst


def download_latest_faprotax(
    target_folder: str | Path = "../../../references/faprotax",
) -> Path:
    """
    Ensure the newest *FAPROTAX.txt* is present locally.

    The file is placed at *<target_folder>/FAPROTAX.txt* (overwriting older ones
    so the wrapper always parses the freshest database).

    Returns
    -------
    Path
        Path to *FAPROTAX.txt*.
    """
    base_url = (
        "https://pages.uoregon.edu/slouca/LoucaLab/archive/"
        "FAPROTAX/lib/php/index.php?section=Download"
    )
    folder = Path(target_folder).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)

    txt_path = folder / "FAPROTAX.txt"
    if txt_path.exists():
        return txt_path  # already cached

    zip_url, version = _scrape_latest_zip_url(base_url)
    zip_name = Path(zip_url.split("?")[0]).name
    zip_path = folder / zip_name

    print(f"Downloading FAPROTAX v{version}: {zip_url}")
    resp = requests.get(zip_url, timeout=120)
    resp.raise_for_status()
    zip_path.write_bytes(resp.content)

    print(f"Extracting FAPROTAX.txt from {zip_name}")
    txt_path = _extract_faprotax_txt(zip_path, folder)

    # keep the ZIP so the version can be inspected later if desired
    return txt_path


def _yield_faprotax_records(fh) -> Iterator[Tuple[str, str]]:
    """Yield *(header, line)* pairs from a *FAPROTAX.txt* file handle."""
    header: str | None = None
    for raw in fh:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(">"):  # trait header
            header = line[1:].strip()
            yield header, "__HEADER__"
        else:
            yield header, line


def _metadata_kv_iter(blob: str) -> Iterator[Tuple[str, str]]:
    """Yield *key*, *value* pairs from the metadata blob of a header line."""
    for part in blob.split():
        if "=" in part:
            k, v = part.split("=", 1)
            yield k.strip(), v.strip()


def parse_faprotax_db(
    path: str | Path, *, compile_regex: bool = True
) -> Dict[
    str, Dict[str, Union[Dict[str, str], List[Dict[str, Union[str, Pattern[str]]]]]]
]:
    """
    Parse *FAPROTAX.txt* into a structured, regex-ready dictionary.

    Returns
    -------
    Dict[trait, Dict]
        ``trait → {"metadata": {...}, "taxa": [{"pat": pattern, "ref": str}]}``
    """
    trait_dict: Dict[
        str, Dict[str, Union[Dict[str, str], List[Dict[str, Union[str, Pattern[str]]]]]]
    ] = {}

    current_trait: str | None = None
    path = Path(path)

    with path.open("rt", encoding="utf-8") as fh:
        for header, line in _yield_faprotax_records(fh):
            if line == "__HEADER__":
                trait, meta_blob = (header.split(maxsplit=1) + [""])[:2]
                trait_dict[trait] = {
                    "metadata": {k: v for k, v in _metadata_kv_iter(meta_blob)},
                    "taxa": [],
                }
                current_trait = trait
                continue

            fields = line.split(None, 1)
            pattern_raw = fields[0]
            ref = (
                fields[1][2:]
                if len(fields) > 1 and fields[1].startswith("//")
                else (fields[1] if len(fields) > 1 else "")
            )

            regex_pat: str | Pattern[str] = f"{pattern_raw.replace('*', '.*')}.*"
            if compile_regex:
                regex_pat = re.compile(regex_pat)

            trait_dict[current_trait]["taxa"].append({"pat": regex_pat, "ref": ref})

    return trait_dict

def find_references_dir(project_name: str = "workflow_16s") -> Path:
    """
    Search upward from the current file or working directory to find the
    '<project_name>/references' folder.

    Parameters
    ----------
    project_name : str
        Name of the root folder containing 'references/'.

    Returns
    -------
    Path
        Path to the references directory.

    Raises
    ------
    FileNotFoundError
        If the directory cannot be found.
    """
    current = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

    for parent in [current] + list(current.parents):
        if parent.name == project_name:
            ref_dir = parent / "references"
            if ref_dir.exists() and ref_dir.is_dir():
                return ref_dir

    raise FileNotFoundError(f"Could not locate '{project_name}/references' directory.")


def get_faprotax_parsed(
    target_folder: str | Path | None = None,
    *,
    compile_regex: bool = True,
):
    """
    Ensure *FAPROTAX.txt* is present and return the parsed dictionary.

    Parameters
    ----------
    target_folder : str | Path | None
        Custom folder where the FAPROTAX database should be stored.
        If None, this function will search for 'workflow_16s/references'.

    Returns
    -------
    dict | None
        Parsed FAPROTAX database or *None* on failure.
    """
    try:
        if target_folder is None:
            target_folder = find_references_dir()
        txt_path = download_latest_faprotax(target_folder)
    except Exception as exc:  # pragma: no cover
        print(f"Failed to obtain FAPROTAX – {exc}")
        return None
    return parse_faprotax_db(txt_path, compile_regex=compile_regex)


def faprotax_functions_for_taxon(
    taxon: str,
    faprotax_db: Dict[
        str,
        Dict[str, Union[Dict[str, str], List[Dict[str, Union[str, Pattern[str]]]]]],
    ],
    *,
    include_references: bool = False,
) -> List[str] | Dict[str, List[str]]:
    """
    Return all FAPROTAX functions (traits) that match ``taxon``.

    Parameters
    ----------
    taxon
        A semicolon-delimited taxonomy string
        (e.g. ``'d__Bacteria;p__Proteobacteria;...'``).
    faprotax_db
        The parsed FAPROTAX dictionary returned by
        :pyfunc:`parse_faprotax_db` or :pyfunc:`get_faprotax_parsed`.
    include_references
        If *True*, return a mapping ``trait → [reference, ...]`` instead of a
        simple list of traits.

    Returns
    -------
    list | dict
        • If *include_references* is *False* (default) → ``[trait, ...]``  
        • Else → ``{trait: [reference, ...], ...}``

    Notes
    -----
    * Matching is **case-insensitive** and performed with
      :pyfunc:`re.search` so that a pattern can match at *any* level of the
      input taxonomy string.
    * Wildcards in FAPROTAX (“*”) are already converted to proper regex
      patterns during parsing.
    """
    taxon_norm = taxon.strip()

    if include_references:
        trait_to_refs: Dict[str, List[str]] = {}
    else:
        traits: List[str] = []

    for trait, entry in faprotax_db.items():
        for rec in entry["taxa"]:
            pat = rec["pat"]  # compiled Pattern[str] (if parse compile_regex=True)
            ref = rec["ref"]

            # Ensure we have a compiled regex (handle compile_regex=False case)
            if isinstance(pat, str):
                pat = re.compile(pat, flags=re.I)

            if pat.search(taxon_norm):
                if include_references:
                    trait_to_refs.setdefault(trait, []).append(ref)
                else:
                    traits.append(trait)
                break  # one pattern match is enough for this trait

    return trait_to_refs if include_references else list(dict.fromkeys(traits))
