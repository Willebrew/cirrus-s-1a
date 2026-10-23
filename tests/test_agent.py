import asyncio

from cirrus.agent.backends import MockBackend
from cirrus.agent.loop import Agent, AgentConfig, ScriptedModel

CORPUS = {
    "https://a.example/foo": ("Foo X1 review", "The Foo X1 e-bike costs $899 and has a 60 km range."),
    "https://b.example/bar": ("Bar Z2 review", "The Bar Z2 e-bike costs $999 with a 80 km range."),
}


def run(turns, cfg=None):
    backend = MockBackend(CORPUS)
    model = ScriptedModel(turns)
    ep = asyncio.run(Agent(model, backend, cfg).run("best budget e-bike"))
    return ep, model, backend


def test_full_episode_with_parallel_search_open_and_answer():
    ep, model, backend = run([
        "<|search|>Foo e-bike<|/search|><|search|>Bar e-bike<|/search|><|run|>",
        "<|open|>1<|/open|><|run|>",
        "<|answer|>The Foo X1 costs $899 with a 60 km range [1]. The Bar Z2 costs $999 [2].<|/answer|>",
    ])
    assert backend.queries == ["Foo e-bike", "Bar e-bike"]
    assert set(ep.sources) == {1, 2}  # deduplicated across both searches
    assert "<|page 1|>" in ep.transcript
    assert ep.audit is not None and ep.audit.ok
    assert ep.rounds == 3
    # second model call saw the rendered results
    assert "<|src 1|>" in model.prompts[1]


def test_forces_answer_when_out_of_rounds():
    ep, _, _ = run(
        ["<|search|>Foo<|/search|><|run|>", "The Foo X1 costs $899 [1].<|/answer|>"],
        AgentConfig(max_rounds=1),
    )
    assert ep.answer == "The Foo X1 costs $899 [1]."


def test_abstention():
    ep, _, _ = run(["<|search|>quantum toaster<|/search|><|run|>", "<|insufficient|>"])
    assert ep.abstained
    assert ep.audit is None


def test_failing_backend_does_not_crash():
    class Broken(MockBackend):
        def search(self, query, k=8):
            raise RuntimeError("boom")

    model = ScriptedModel(["<|search|>x<|/search|><|run|>", "<|insufficient|>"])
    ep = asyncio.run(Agent(model, Broken({})).run("x"))
    assert ep.abstained and ep.sources == {}
