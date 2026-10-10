# Memory v2.1 r5: the three replay arms as switch presets

Design: research repo `docs/design/memory-v2.1-writer-r5.md` §6. Every arm runs with `UNIFY_MEMORY_V21=on` and the
same S0, shared-text markers, identical-part credit, finish G1 check, write-phase repair pricing, S2 replay
verification, S3/S4 and S6 limits. They differ only in who reads each episode before Sol writes.

| arm | `UNIFY_MEMORY_V21_FORK` | `UNIFY_MEMORY_V21_ANALYSTS` | staging comes from |
|---|---|---|---|
| A: single writer | off | off | nothing (Sol reads with scan-first, the overview and the markers) |
| B: Luna fork + Sol merge | on (live runs) | off | the actor's episode-end fork (`fork.json`); in the replay, the arm-B driver (`replay_fork.json`) |
| C: Sol analysts + Sol merge | off | sol | one Sol analyst per flagged episode (`analyst.json`) |

Spend by category: the writer and the analysts are in each pass's end event (`usd_by_category`); the fork's calls
are in the proxy journal under `unify_meta.session = fork.<episode>` (call kind `actor_fork`, journaled `other` by
the pinned keyproxy). A v2.1 pass is uncapped only with a Sol proxy route (`UNIFY_MEMORY_V2_SOL_BASE_URL`, which
must point at a proxy with a spend ceiling).
