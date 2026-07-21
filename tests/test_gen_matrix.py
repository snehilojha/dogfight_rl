from evaluation.gen_matrix import _acceptance


def _matrix_from(pairs, gens):
    """Build a full matrix dict from a {(ego, opp): win_rate} mapping."""
    m = {i: {j: (None if i == j else pairs.get((i, j))) for j in gens} for i in gens}
    return m


def test_acceptance_passes_when_later_beats_earlier():
    gens = [0, 1, 2, 3, 4]
    # Later generations dominate earlier ones.
    pairs = {(3, 0): 1.0, (4, 1): 0.8}
    matrix = _matrix_from(pairs, gens)
    passed, checks = _acceptance(gens, matrix, gap=3, threshold=0.6)
    assert passed
    assert {c["later"] for c in checks} == {3, 4}
    assert all(c["pass"] for c in checks)


def test_acceptance_fails_when_below_threshold():
    gens = [0, 1, 2, 3]
    pairs = {(3, 0): 0.4}
    matrix = _matrix_from(pairs, gens)
    passed, checks = _acceptance(gens, matrix, gap=3, threshold=0.6)
    assert not passed
    assert len(checks) == 1
    assert checks[0]["pass"] is False


def test_acceptance_empty_when_pool_too_small():
    gens = [0, 1]
    matrix = _matrix_from({}, gens)
    passed, checks = _acceptance(gens, matrix, gap=3, threshold=0.6)
    # No gen is 3 apart from another, so there is nothing to check → not passed.
    assert checks == []
    assert not passed
