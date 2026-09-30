"""
search terms are safed in dzg_search_terms.yaml.

A term is normally matched as written, anywhere in the affiliation and ignoring case.
Three markers at the start of a term change that:

  {noorder}   every word after the marker has to occur somewhere in the affiliation,
              in any order.

  {nosearch}  the term is not used for the search in PubMed because it doesn`t proof membership in the DZG.
              Use it for wording that is too general, which would return many unrelated articles.
              It only fills the site column.

  {regex}     the rest of the entry is a regular expression, matched ignoring case. Use it
              when a site is only recognisable from the wording around it, such as an anchor
              word next to a city name. PubMed cannot search with a pattern, so the term
              behaves like {nosearch} and fills the site column only. Combining it with
              {noorder} is rejected.
"""

import re
from typing import NamedTuple

import pandas as pd

KNOWN_MARKERS = {"noorder", "nosearch", "regex"}


class SearchTerm(NamedTuple):
    """One entry of a search term list, with its markers already read."""

    words: list[str]            # all of these have to occur, empty for a pattern
    pattern: re.Pattern | None  # set for {regex}, then words is empty
    searchable: bool            # whether the term may be sent to PubMed


def parse_term(raw: str) -> SearchTerm:
    """Split one YAML entry into its markers and the words to look for.
    Raises ValueError on an unknown marker.
    """
    text = raw.strip()
    markers = set()

    while text.startswith("{"):
        marker, closing, rest = text.partition("}")
        if not closing:
            raise ValueError(f"Search term {raw!r} has an opening brace without a closing one")
        markers.add(marker[1:].strip().lower())
        text = rest.strip()

    unknown = markers - KNOWN_MARKERS
    if unknown:
        raise ValueError(
            f"Search term {raw!r} uses unknown marker(s) {sorted(unknown)}. "
            f"Known markers are {sorted(KNOWN_MARKERS)}."
        )
    if not text:
        raise ValueError(f"Search term {raw!r} has markers but no words to search for")

    if "regex" in markers:
        if "noorder" in markers:
            raise ValueError(f"Search term {raw!r} combines {{regex}} with {{noorder}}. "
                             "Splitting a pattern at its spaces would break it.")
        try:
            pattern = re.compile(text, re.IGNORECASE)
        except re.error as error:
            raise ValueError(f"Search term {raw!r} is not a valid regular expression: {error}")
        return SearchTerm(words=[], pattern=pattern, searchable=False)

    words = text.split() if "noorder" in markers else [text]
    return SearchTerm(words=words, pattern=None, searchable="nosearch" not in markers)


def dzg_terms(networks: dict) -> list[str]:
    """Collect the terms of one DZG that identify the DZG itself.
    Runs over the general block and every site block. {nosearch} and {regex} terms are
    skipped.
    """
    terms = []
    for network in networks.values():
        for raw in network:
            if parse_term(raw).searchable:
                terms.append(raw)
    return terms


def build_query(networks: dict) -> str:
    """Build the PubMed affiliation query for one DZG.
    Every term becomes one clause and the clauses are joined with OR. A {noorder} term
    carries several words, those become an AND group in brackets so that PubMed only
    returns affiliations containing all of them.
    """
    clauses = []
    for raw in dzg_terms(networks):
        words  = parse_term(raw).words
        fields = [f'"{word}"[Affiliation]' for word in words]
        if len(fields) == 1:
            clauses.append(fields[0])
        else:
            clauses.append("(" + " AND ".join(fields) + ")")
    return " OR ".join(clauses)


def affiliation_matches(affiliations: pd.Series, terms: list[str]) -> pd.Series:
    """Mark the affiliations that match any of the search terms.
    A plain term matches when all of its words occur in the affiliation, a {regex} term
    when its pattern is found anywhere. Both ignore case.
    """
    word_groups, patterns = [], []
    for raw in terms:
        term = parse_term(raw)
        if term.pattern is not None:
            patterns.append(term.pattern)
        else:
            word_groups.append([word.lower() for word in term.words])

    def matches_any_term(affiliation: str) -> bool:
        original = str(affiliation)
        lowered = original.lower()
        for words in word_groups:
            if all(word in lowered for word in words):
                return True
        return any(pattern.search(original) for pattern in patterns)

    return affiliations.apply(matches_any_term)