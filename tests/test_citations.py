from cirrus.agent.citations import (
    CitationConstraint,
    audit_answer,
    cited_ids,
    lexical_support,
)


def test_cited_ids():
    assert cited_ids("x [1] y [2, 13] z") == [1, 2, 13]


def test_audit_flags_uncited_and_unknown():
    ans = "## Summary\nThe Foo X1 costs $899 at launch [1]. It weighs about twenty two kilograms total. Bar [7]."
    a = audit_answer(ans, {1, 2})
    assert a.unknown_ids == [7]
    assert a.uncited_claims == ["It weighs about twenty two kilograms total."]
    assert 0 < a.citation_coverage < 1
    assert not a.ok


def test_lexical_support_weights_numbers():
    sources = {1: "The Foo X1 launched at $899 in September 2026."}
    good = lexical_support("The Foo X1 launched at $899 [1].", sources)
    bad = lexical_support("The Foo X1 launched at $1299 [1].", sources)
    assert good == 1.0
    assert bad < good
    assert lexical_support("No citation here.", sources) == 0.0


def test_constraint_only_allows_valid_ids():
    c = CitationConstraint({3, 12, 15})
    assert c.allowed_next("Hello") is None
    assert c.allowed_next("Hello [") == {"3", "1"}
    assert c.allowed_next("Hello [1") == {"2", "5"}
    assert c.allowed_next("Hello [12") == {"]", ","}
    assert c.allowed_next("Hello [3] done") is None
    assert "1" in c.allowed_next("Hello [3,")
