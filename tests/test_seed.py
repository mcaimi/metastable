from metastable import seed


def test_random_seed_respects_bit_length():
    for _ in range(100):
        value = seed.get_random_seed(8)
        assert 0 <= value < 2**8


def test_default_bit_length_is_32():
    assert seed.RANDOM_BIT_LENGTH == 32
    assert 0 <= seed.get_random_seed() < 2**32


def test_resolve_seed_keeps_non_negative_values():
    assert seed.resolve_seed(0) == 0
    assert seed.resolve_seed(1234) == 1234


def test_resolve_seed_randomizes_negative_values(monkeypatch):
    monkeypatch.setattr(seed, "get_random_seed", lambda bits: 777)
    assert seed.resolve_seed(-1) == 777
    assert seed.resolve_seed(-42) == 777
