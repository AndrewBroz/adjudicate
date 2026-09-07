from adjudicate.dictionary import known, known_anywhere


def test_lookups_with_inflection_and_case():
    assert known("travelling", "uk") and not known("travelling", "us")
    assert known("Colour", "uk") and known("honored", "us")
    assert known_anywhere("cyberattack") and not known_anywhere("platformize")
