from runtime.token_metrics import TokenMetrics


def test_token_metrics_initial_zeros():
    m = TokenMetrics("input", "output", "requests", "work")
    rendered = m.render()
    assert "aipod:prompt_tokens_total 0" in rendered
    assert "aipod:generation_tokens_total 0" in rendered
    assert "aipod:requests_total 0" in rendered
    assert "aipod:work_total 0" in rendered


def test_token_metrics_monotonic_and_invalid_handling():
    m = TokenMetrics("input", "output", "requests", "work")
    m.add(input=10, output=20, requests=1, work=1.5)
    m.add(input=5, output=8, requests=1, work=2.0)
    rendered = m.render()
    assert "aipod:prompt_tokens_total 15" in rendered
    assert "aipod:generation_tokens_total 28" in rendered
    assert "aipod:requests_total 2" in rendered
    assert "aipod:work_total 3.5" in rendered
    # Invalid negative value marks that metric unknown (omitted, not zeroed)
    m.add(input=-1)
    assert "prompt_tokens_total" not in m.render()
    assert "aipod:generation_tokens_total 28" in m.render()
