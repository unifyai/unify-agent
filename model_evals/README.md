# Evals of a live model (model_evals/)

These measure what a real model does: whether the compactor keeps what matters across passes and images, whether a
provider enforces a required tool choice, whether the model re-issues a refused call with the parameters named, and
how it handles time. Each needs a real provider key and spends money, so none is part of the test suite. The suite
(`tests/`) never reaches a model provider or the network: its models are scripted (`tests/scripted_model.py`).

Nothing here is collected by a plain `pytest` (`pytest.ini` sets `testpaths = tests`, and these files are named
`eval_*.py`). Run one by hand, with a key loaded, from the repository root:

```bash
pytest -o python_files='eval_*.py' -p no:cacheprovider model_evals/eval_tool_choice.py
```

| file | what it measures | moved from |
|---|---|---|
| `eval_context_compression.py` | multi-pass and image-aware compaction quality | `tests/async_tool_loop/test_context_compression.py` |
| `eval_tool_choice.py` | the provider enforces `tool_choice="required"` against a contrary instruction | `tests/async_tool_loop/test_tool_loop_limits.py` |
| `eval_refused_call_retry.py` | the model reads a refusal and re-issues the call with the parameters named | `tests/async_tool_loop/test_unknown_arguments.py` |
| `eval_time_awareness.py` | the model's use of the time it is given | `tests/async_tool_loop/test_time_awareness_eval.py` |
| `eval_store_review_judgement.py` | the storage review judges a function worth storing | `tests/actor/code_act/test_can_compose_and_store.py` |
| `legacy/eval_act_llm_profile.py` | a request for maximum thinking picks the high-effort profile | `tests/legacy/conversation_manager/core/test_cm_act_llm_profile.py` |
| `legacy/eval_multilingual_replies.py` | the model replies in the user's language and acts in English | `tests/legacy/conversation_manager/flows/test_multilingual.py` (its tests now check that the harness carries the language both ways) |

The legacy evals take the legacy conversation manager's own test fixtures (`model_evals/legacy/conftest.py`).
