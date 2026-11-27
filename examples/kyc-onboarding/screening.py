"""Sanctions screening: the deterministic half. It finds the candidates; the agent judges them.

Two kinds of lists are searched:

- **Sanctions lists.** By default a fictional list in the format of the EU consolidated list
  (``lists/fictional_eu_list.csv``), with made-up entries, so the example runs offline and
  gives the same result every time. With ``SANCTIONS_LISTS=official`` the real lists are
  used instead: the EU consolidated list and the US OFAC SDN list, downloaded from their
  official sources on first use and kept for a day in your system's temporary folder.
- **The bank's internal watchlist** (``lists/internal_watchlist.csv``), fictional too: parties
  the bank has already decided not to deal with. Searched in both modes.

Only list files are ever downloaded: the names screened never leave this machine.

A name matches when it is close enough to a listed name or alias, after normalising both
(case, accents, punctuation, word order, legal suffixes such as "Srl" or "Ltd"). Each
candidate comes with what the list says about it (birth dates, countries, programme), and
whether those agree with what was screened, for the agent to decide.
"""

import csv
import io
import os
import re
import tempfile
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from functools import cache
from pathlib import Path

HERE = Path(__file__).parent
FICTIONAL_LIST = HERE / "lists" / "fictional_eu_list.csv"
INTERNAL_WATCHLIST = HERE / "lists" / "internal_watchlist.csv"

EU_URL = (
    "https://webgate.ec.europa.eu/fsd/fsf/public/files/csvFullSanctionsList_1_1/content"
    "?token=dG9rZW4tMjAxNw"  # the public token the EU's data portal gives for this file
)
OFAC_URL = "https://sanctionslistservice.ofac.treas.gov/api/PublicationPreview/exports/{}"
CACHE = Path(tempfile.gettempdir()) / "domyn_kyc_lists"
CACHE_SECONDS = 24 * 3600

MATCH_THRESHOLD = 0.85  # how close a name must be to be a candidate
MAX_CANDIDATES = 5

# Words that say what a company is, not which one: "Rossi Srl" and "Rossi S.r.l." are one name.
LEGAL_FORMS = {
    "srl", "spa", "sas", "snc", "sapa", "ltd", "limited", "llc", "inc", "plc", "gmbh",
    "ag", "sa", "sarl", "bv", "nv", "co", "corp", "company", "the",
}  # fmt: skip


@dataclass
class Entry:
    """One listed party, with every name it is listed under."""

    source: str
    entry_id: str
    kind: str  # person or entity
    names: list[str] = field(default_factory=list)
    birth_dates: set[str] = field(default_factory=set)
    countries: set[str] = field(default_factory=set)
    programme: str = ""
    listed_on: str = ""
    remark: str = ""


# --- Reading the lists ---------------------------------------------------------------------


def _read_eu(text: str, source: str) -> list[Entry]:
    """The EU consolidated list: one row per name, birth date, citizenship… of an entry."""
    entries: dict[str, Entry] = {}
    for row in csv.DictReader(io.StringIO(text.lstrip("﻿")), delimiter=";"):
        entry_id = row["Entity_LogicalId"]
        entry = entries.get(entry_id)
        if entry is None:
            kind = "person" if row["Entity_SubjectType"] == "P" else "entity"
            entry = entries[entry_id] = Entry(
                source=source,
                entry_id=f"EU-{entry_id}",
                kind=kind,
                programme=row["Entity_Regulation_Programme"],
                listed_on=row["Entity_DesignationDate"] or row["Entity_Regulation_PublicationDate"],
                remark=row["Entity_Remark"],
            )
        if (name := row["NameAlias_WholeName"].strip()) and name not in entry.names:
            entry.names.append(name)
        if born := row["BirthDate_BirthDate"].strip():
            entry.birth_dates.add(born)
        elif year := row["BirthDate_Year"].strip():
            entry.birth_dates.add(year)
        for country in (
            row["Citizenship_CountryIso2Code"],
            row["BirthDate_CountryIso2Code"],
            row["Address_CountryIso2Code"],  # an entity's only country
        ):
            if country.strip():
                entry.countries.add(country.strip().upper())
    return list(entries.values())


_DOB = re.compile(r"DOB (?:circa )?(\d{1,2} \w{3} \d{4}|\w{3} \d{4}|\d{4})")
_NATIONALITY = re.compile(r"(?:nationality|citizen) ([A-Za-z ]+?)[;.]")


def _read_ofac(sdn: str, alt: str) -> list[Entry]:
    """The OFAC SDN list: names in ``SDN.CSV``, aliases in ``ALT.CSV``, dates in the remarks."""
    entries: dict[str, Entry] = {}
    for row in csv.reader(io.StringIO(sdn)):
        if len(row) < 12 or row[2].strip() in ("vessel", "aircraft"):
            continue
        remarks = row[11] if row[11].strip() != "-0-" else ""
        entries[row[0]] = Entry(
            source="OFAC SDN",
            entry_id=f"OFAC-{row[0]}",
            kind="person" if row[2].strip() == "individual" else "entity",
            names=[_ofac_name(row[1])],
            birth_dates={iso_date(d) for d in _DOB.findall(remarks)},
            countries={c.strip() for c in _NATIONALITY.findall(remarks)},
            programme=row[3].strip(),
            remark=remarks[:300],
        )
    for row in csv.reader(io.StringIO(alt)):
        if len(row) >= 4 and row[0] in entries:
            entries[row[0]].names.append(_ofac_name(row[3]))
    return list(entries.values())


def _ofac_name(name: str) -> str:
    """OFAC writes people as "LAST, First": put them in reading order."""
    last, _, first = name.partition(", ")
    return f"{first} {last}".strip() if first else name.strip()


def iso_date(date: str) -> str:
    """An OFAC date (``10 Dec 1948``, ``Dec 1948``, ``1948``) as ISO, as far as it goes."""
    for pattern, iso in (("%d %b %Y", "%Y-%m-%d"), ("%b %Y", "%Y-%m"), ("%Y", "%Y")):
        try:
            return datetime.strptime(date, pattern).strftime(iso)
        except ValueError:
            continue
    return date


def _read_watchlist(path: Path) -> list[Entry]:
    entries = []
    for row in csv.DictReader(path.open(encoding="utf-8"), delimiter=";"):
        entries.append(
            Entry(
                source="Internal watchlist",
                entry_id=row["id"],
                kind=row["kind"],
                names=[row["name"], *filter(None, row["aliases"].split("|"))],
                birth_dates=set(filter(None, [row["birth_date"]])),
                countries=set(filter(None, [row["country"]])),
                listed_on=row["added"],
                remark=row["reason"],
            )
        )
    return entries


def _download(url: str, name: str) -> str:
    """A list file from its official source, kept for a day."""
    import httpx

    path = CACHE / name
    if not path.exists() or time.time() - path.stat().st_mtime > CACHE_SECONDS:
        CACHE.mkdir(parents=True, exist_ok=True)
        response = httpx.get(url, timeout=120, follow_redirects=True)
        response.raise_for_status()
        path.write_bytes(response.content)
    return path.read_text(encoding="utf-8")


def mode() -> str:
    return os.getenv("SANCTIONS_LISTS", "fictional").strip().lower()


@cache
def lists() -> tuple[list[Entry], list[dict[str, str]]]:
    """Every entry searched, and which lists they come from."""
    watchlist = _read_watchlist(INTERNAL_WATCHLIST)
    if mode() == "official":
        eu = _read_eu(_download(EU_URL, "eu_consolidated.csv"), "EU consolidated list")
        ofac = _read_ofac(
            _download(OFAC_URL.format("SDN.CSV"), "ofac_sdn.csv"),
            _download(OFAC_URL.format("ALT.CSV"), "ofac_alt.csv"),
        )
        sources = [
            {"list": "EU consolidated list", "entries": str(len(eu)), "source": "official"},
            {"list": "OFAC SDN", "entries": str(len(ofac)), "source": "official"},
        ]
        listed = eu + ofac
    else:
        listed = _read_eu(FICTIONAL_LIST.read_text(encoding="utf-8"), "Fictional EU-format list")
        sources = [
            {
                "list": "Fictional EU-format list",
                "entries": str(len(listed)),
                "source": "made up for this example: set SANCTIONS_LISTS=official for the real lists",
            }
        ]
    sources.append(
        {"list": "Internal watchlist", "entries": str(len(watchlist)), "source": "the bank's own"}
    )
    return listed + watchlist, sources


# --- Matching -------------------------------------------------------------------------------


def tokens(name: str) -> list[str]:
    """A name as comparable words: no accents, no punctuation, no legal form, in order."""
    folded = unicodedata.normalize("NFKD", name)
    folded = "".join(c for c in folded if not unicodedata.combining(c)).lower()
    folded = folded.replace(".", "")  # "S.r.l." is "srl"
    words = re.sub(r"[^a-z0-9]+", " ", folded).split()
    return sorted(w for w in words if w not in LEGAL_FORMS)


def similarity(a: list[str], b: list[str]) -> float:
    """How alike two names are, from 0 to 1, whatever the order of their words."""
    if not a or not b:
        return 0.0
    whole = SequenceMatcher(None, " ".join(a), " ".join(b)).ratio()
    # Each word of the shorter name against its closest word in the longer one: "Mario
    # Rossi" in "Mario Giuseppe Rossi" is a strong match the whole-string ratio undersells.
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    words = sum(max(SequenceMatcher(None, w, v).ratio() for v in long) for w in short)
    return max(whole, words / len(short) * min(1.0, (len(short) + 1) / len(long)))


@cache
def _index() -> dict[str, list[tuple[Entry, str, list[str]]]]:
    """Listed names by the first three letters of each of their words, to compare few."""
    index: dict[str, list[tuple[Entry, str, list[str]]]] = {}
    for entry in lists()[0]:
        for name in entry.names:
            words = tokens(name)
            for prefix in {w[:3] for w in words}:
                index.setdefault(prefix, []).append((entry, name, words))
    return index


def find_candidates(name: str, kind: str = "any") -> list[tuple[float, Entry, str]]:
    """Listed parties whose name is close to ``name``, best first: (score, entry, name)."""
    query = tokens(name)
    best: dict[str, tuple[float, Entry, str]] = {}
    for prefix in {w[:3] for w in query}:
        for entry, listed, words in _index().get(prefix, []):
            if kind in ("person", "entity") and entry.kind != kind:
                continue
            score = similarity(query, words)
            key = f"{entry.source}/{entry.entry_id}"
            if score >= MATCH_THRESHOLD and score > best.get(key, (0.0,))[0]:
                best[key] = (score, entry, listed)
    return sorted(best.values(), key=lambda c: -c[0])[:MAX_CANDIDATES]


def _agrees(screened: str, listed: set[str]) -> bool | None:
    """Whether a birth date or country screened agrees with the list: None when either is
    unknown, or when the list writes it another way (OFAC names countries, "Russia", where
    the EU list and the screening use ISO codes, "RU"). A year agrees with a full date in
    that year."""
    if screened.isalpha():  # a country code: compare with codes only
        listed = {v for v in listed if len(v) == 2 and v.isalpha()}
    if not screened or not listed:
        return None
    return any(screened.startswith(v) or v.startswith(screened) for v in listed)


# --- The tool -------------------------------------------------------------------------------


def screen_name(name: str, kind: str = "any", birth_date: str = "", country: str = "") -> dict:
    """Screen one name against the sanctions lists and the bank's internal watchlist.

    Give a person's birth date (YYYY-MM-DD) and citizenship (ISO code such as IT) when you
    know them: each candidate then says whether they agree with the list. ``kind`` is
    "person", "entity" or "any". The search is by name only and finds candidates, not
    matches: deciding whether a candidate is the same party is your job.
    """
    candidates = []
    for score, entry, listed in find_candidates(name, kind):
        candidates.append(
            {
                "list": entry.source,
                "entry_id": entry.entry_id,
                "kind": entry.kind,
                "listed_name": listed,
                "other_names": [n for n in entry.names if n != listed][:5],
                "name_score": round(score, 2),
                "birth_dates": sorted(entry.birth_dates),
                "countries": sorted(entry.countries),
                "programme": entry.programme,
                "listed_on": entry.listed_on,
                "remark": entry.remark,
                "birth_date_agrees": _agrees(birth_date, entry.birth_dates),
                "country_agrees": _agrees(country.upper(), entry.countries),
            }
        )
    return {"screened": name, "candidates": candidates, "lists_searched": lists()[1]}
