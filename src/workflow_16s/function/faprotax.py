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

def find_references_dir(project_name: str = "workflow_16s") -> Path:
    """
    Walk upward from CWD (or script dir) to locate '<project_name>/references'.

    Returns
    -------
    Path
        Absolute path to the references directory.

    Raises
    ------
    FileNotFoundError
        If the directory cannot be found.
    """
    start = (
        Path(__file__).resolve().parent
        if "__file__" in globals()
        else Path.cwd().resolve()
    )

    for parent in [start] + list(start.parents):
        if parent.name == project_name:
            ref_dir = parent / "references"
            if ref_dir.is_dir():
                return ref_dir

    raise FileNotFoundError(
        f"Could not locate '{project_name}/references' directory starting from {start}"
    )


# -----------------------------------------------------------------------------#
# Download helpers
# -----------------------------------------------------------------------------#


def _scrape_latest_zip_url(base_url: str) -> Tuple[str, str]:
    """Return (zip_url, version_str) for the newest *FAPROTAX_*.zip."""
    soup = BeautifulSoup(requests.get(base_url, timeout=30).text, "html.parser")

    zip_links: list[Tuple[str, Tuple[int, ...]]] = []
    rgx = re.compile(r"FAPROTAX_(\d+(?:\.\d+)*)\.zip$", re.I)

    for a in soup.find_all("a", href=True):
        if (m := rgx.search(a["href"])) is not None:
            version_tuple = tuple(int(p) for p in m.group(1).split("."))
            zip_links.append((urljoin(base_url, a["href"]), version_tuple))

    if not zip_links:
        raise RuntimeError("No FAPROTAX *.zip links found on the download page.")

    zip_links.sort(key=lambda x: x[1], reverse=True)
    url, ver = zip_links[0]
    return url, ".".join(map(str, ver))


def _extract_faprotax_txt(zip_path: Path, out_folder: Path) -> Path:
    """Extract FAPROTAX.txt from *zip_path* into *out_folder* and return its path."""
    with zipfile.ZipFile(zip_path) as zf:
        txt_members = [m for m in zf.namelist() if m.endswith("FAPROTAX.txt")]
        if not txt_members:
            raise RuntimeError("FAPROTAX.txt not found inside ZIP archive.")
        extracted = zf.extract(txt_members[0], path=out_folder)
        dst = out_folder / "FAPROTAX.txt"
        Path(extracted).replace(dst)  # overwrite / standardise name
        return dst


def download_latest_faprotax(destination: Path) -> Path:
    """
    Ensure *FAPROTAX.txt* in **destination**/<file>. Creates *destination*.

    Parameters
    ----------
    destination : Path
        Folder where FAPROTAX files should reside (e.g. .../references/faprotax).

    Returns
    -------
    Path
        Path to the up-to-date *FAPROTAX.txt*.
    """
    base_url = (
        "https://pages.uoregon.edu/slouca/LoucaLab/archive/"
        "FAPROTAX/lib/php/index.php?section=Download"
    )
    destination.mkdir(parents=True, exist_ok=True)

    txt_path = destination / "FAPROTAX.txt"
    if txt_path.exists():
        return txt_path  # already cached

    zip_url, version = _scrape_latest_zip_url(base_url)
    zip_path = destination / Path(zip_url.split("?")[0]).name

    logging.info("Downloading FAPROTAX v%s from %s", version, zip_url)
    zip_path.write_bytes(requests.get(zip_url, timeout=120).content)

    logging.info("Extracting FAPROTAX.txt to %s", destination)
    return _extract_faprotax_txt(zip_path, destination)


# -----------------------------------------------------------------------------#
# Parsing helpers
# -----------------------------------------------------------------------------#


def _yield_faprotax_records(fh) -> Iterator[Tuple[str, str]]:
    """
    Yield pairs of (header, line) from the FAPROTAX file handle.

    Header lines do NOT start with '>' in your format.
    Instead, treat lines containing metadata (like ';' or key:value pairs) as headers.
    """
    header: str | None = None
    for raw in fh:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        # Heuristic for header line:
        # If the line contains ';' or expected metadata keys and no leading spaces, treat as header
        if (
            (";" in line or "elements:" in line or "exclusively_prokaryotic" in line)
            and not raw.startswith(" ")
        ):
            header = line
            yield header, "__HEADER__"
        else:
            if header is None:
                # Skip lines before any header (could log warning here)
                continue
            yield header, line


def _metadata_kv_iter(blob: str) -> Iterator[Tuple[str, str]]:
    """
    Parse metadata blob into key, value pairs.
    Metadata blob is expected to be a string like:
    "elements:C,H; main_element:C; electron_donor:C; electron_acceptor:variable; ..."
    """
    # Split by semicolon or whitespace, then key:value or key=value
    for part in re.split(r"[;\s]+", blob):
        if not part.strip():
            continue
        if ":" in part:
            k, v = part.split(":", 1)
            yield k.strip(), v.strip()
        elif "=" in part:
            k, v = part.split("=", 1)
            yield k.strip(), v.strip()
        else:
            # No key-value separator; yield as key with empty value
            yield part.strip(), ""


def parse_faprotax_db(
    path: str | Path,
    *,
    compile_regex: bool = True,
) -> Dict[
    str,
    Dict[str, Union[Dict[str, str], List[Dict[str, Union[str, Pattern[str]]]]]],
]:
    """
    Parse a FAPROTAX.txt file into a structured, regex-ready dictionary.

    Parameters
    ----------
    path : str | Path
        Path to FAPROTAX.txt file.
    compile_regex : bool
        If True, compile patterns with re.compile() for faster matching.

    Returns
    -------
    dict
        Mapping from trait name to dict with metadata and taxa patterns.
    """
    trait_dict: Dict[
        str,
        Dict[str, Union[Dict[str, str], List[Dict[str, Union[str, Pattern[str]]]]]],
    ] = {}

    current_trait: str | None = None
    path = Path(path)

    with path.open(encoding="utf-8") as fh:
        for header, line in _yield_faprotax_records(fh):
            if line == "__HEADER__":
                trait, meta_blob = (header.split(maxsplit=1) + [""])[:2]
                trait_dict[trait] = {
                    "metadata": dict(_metadata_kv_iter(meta_blob)),
                    "taxa": [],
                }
                current_trait = trait
                continue

            if current_trait is None:
                raise ValueError(
                    f"Found FAPROTAX pattern line before any trait header: {line}"
                )

            fields = line.split(None, 1)  # pattern [reference]
            pattern_raw = fields[0]
            ref = (
                fields[1][2:]
                if len(fields) > 1 and fields[1].startswith("//")
                else (fields[1] if len(fields) > 1 else "")
            )

            regex_pat: str | Pattern[str] = pattern_raw.replace("*", ".*") + ".*"
            if compile_regex:
                regex_pat = re.compile(regex_pat)

            trait_dict[current_trait]["taxa"].append({"pat": regex_pat, "ref": ref})

    return trait_dict



# -----------------------------------------------------------------------------#
# Public convenience wrapper
# -----------------------------------------------------------------------------#


def get_faprotax_parsed(
    *,
    compile_regex: bool = True,
) -> Dict[
    str,
    Dict[str, Union[Dict[str, str], List[Dict[str, Union[str, Pattern[str]]]]]],
] | None:
    """
    Locate workflow_16s/references/, ensure a faprotax/ subdir, download /
    parse the latest FAPROTAX database, and return it.

    Returns
    -------
    dict | None
        Parsed FAPROTAX or *None* on failure.
    """
    try:
        references_dir = find_references_dir()
        faprotax_dir = references_dir / "faprotax"
        faprotax_txt = download_latest_faprotax(faprotax_dir)
        return parse_faprotax_db(faprotax_txt, compile_regex=compile_regex)
    except Exception as exc:
        import traceback

        logger.exception(
            "Failed to prepare FAPROTAX database: %s\n%s",
            exc,
            traceback.format_exc(),
        )
        return None


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
