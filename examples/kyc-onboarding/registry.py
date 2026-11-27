"""The business registry: who runs and who owns a company. Made-up companies and people.

A stand-in for a national company registry (in Italy, the Registro delle Imprese). Every
company, person and number here is fictional.
"""

import re

COMPANIES = {
    "Lanterna Meccanica Srl": {
        "vat_number": "IT10000000001",
        "registered": "2014-05-12",
        "address": "Via del Molo 18, Genova, IT",
        "activity": "Precision machining of metal parts for shipyards",
        "directors": [
            {
                "name": "Giada Ferlandi",
                "role": "sole director",
                "birth_date": "1979-02-11",
                "citizenship": "IT",
            },
        ],
        "shareholders": [
            {"name": "Giada Ferlandi", "kind": "person", "share": 100.0},
        ],
    },
    "Velarte Design Srl": {
        "vat_number": "IT10000000002",
        "registered": "2019-09-03",
        "address": "Corso Garibaldi 71, Milano, IT",
        "activity": "Interior design studio",
        "directors": [
            {
                "name": "Lorenzo Vartani",
                "role": "managing director",
                "birth_date": "1988-03-14",
                "citizenship": "IT",
            },
        ],
        "shareholders": [
            {"name": "Lorenzo Vartani", "kind": "person", "share": 60.0},
            {"name": "Elena Brusati", "kind": "person", "share": 40.0},
        ],
    },
    "Nordavia Logistics Srl": {
        "vat_number": "IT10000000003",
        "registered": "2023-02-20",
        "address": "Riva Nazario Sauro 4, Trieste, IT",
        "activity": "Freight forwarding by sea and road",
        "directors": [
            {
                "name": "Paolo Gremaldi",
                "role": "sole director",
                "birth_date": "1972-06-30",
                "citizenship": "IT",
            },
        ],
        "shareholders": [
            {"name": "Halvern Trade Holdings Ltd", "kind": "company", "share": 70.0},
            {"name": "Paolo Gremaldi", "kind": "person", "share": 30.0},
        ],
    },
    "Halvern Trade Holdings Ltd": {
        "vat_number": None,
        "registered": "2021-08-11",
        "address": "London, GB (foreign company: from its own registry's extract)",
        "activity": "Holding company",
        "directors": [
            {
                "name": "Ardan Velys",
                "role": "director",
                "birth_date": "1970-01-15",
                "citizenship": "CY",
            },
        ],
        "shareholders": [
            {"name": "Tarkenvale Maritime Group LLC", "kind": "company", "share": 100.0},
        ],
    },
    "Studio Calvari Associati Srl": {
        "vat_number": "IT10000000004",
        "registered": "2022-01-17",
        "address": "Via Santo Stefano 40, Bologna, IT",
        "activity": "Tax and accounting advisory",
        "directors": [
            {
                "name": "Marta Calvari",
                "role": "chair",
                "birth_date": "1984-10-02",
                "citizenship": "IT",
            },
            {
                "name": "Andrea Siroli",
                "role": "director",
                "birth_date": "1981-04-23",
                "citizenship": "IT",
            },
        ],
        "shareholders": [
            {"name": "Marta Calvari", "kind": "person", "share": 50.0},
            {"name": "Andrea Siroli", "kind": "person", "share": 50.0},
        ],
    },
}


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def find_company(name_or_vat: str) -> tuple[str, dict] | None:
    """A company by name (any spelling of case and punctuation) or by VAT number."""
    wanted = _key(name_or_vat)
    for name, company in COMPANIES.items():
        if wanted in (_key(name), _key(company["vat_number"] or "")):
            return name, company
    return None


def lookup_company(name_or_vat: str) -> dict | str:
    """Look up a company in the business registry by name or VAT number: its registration,
    activity, directors (with birth date and citizenship) and shareholders (people or
    companies, with their share). A shareholder that is a company can be looked up in turn."""
    found = find_company(name_or_vat)
    if found is None:
        return f"No company {name_or_vat!r} in the registry."
    name, company = found
    return {"name": name, **company}


def parties(name_or_vat: str, depth: int = 4) -> list[dict]:
    """Every party behind a company: the company, its directors, and its shareholders,
    following shareholders that are companies down the chain. For the bank's rules."""
    found = find_company(name_or_vat)
    if found is None:
        return []
    name, company = found
    result = [{"name": name, "kind": "entity"}]
    for director in company["directors"]:
        result.append({"name": director["name"], "kind": "person", **director})
    for holder in company["shareholders"]:
        kind = "person" if holder["kind"] == "person" else "entity"
        result.append({"name": holder["name"], "kind": kind})
        if kind == "entity" and depth > 0:
            result += parties(holder["name"], depth - 1)[1:]
    seen, unique = set(), []
    for party in result:
        if (party["name"], party["kind"]) not in seen:
            seen.add((party["name"], party["kind"]))
            unique.append(party)
    return unique
