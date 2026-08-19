# Corrections

Claims made during this study that later measurement disproved. Kept so a reader of the
git history knows which conclusions were withdrawn and what replaced them.

| claim | status | replaced by |
|---|---|---|
| H4D cannot offer `FI_RMA` and `FI_ATOMIC` on one endpoint | wrong; artefact of `fi_info -c` not parsing comma-separated capabilities | `results/h4d-psm3.md` |
| Symmetric heap caps at about 2 GB per PE | wrong; tests confounded per-PE with per-node. The real ceiling is per-node aggregate | `windowed-exchange.md` |
| The instability is `FI_DELIVERY_COMPLETE` or TX queue exhaustion | wrong; it is connection establishment | `rxm-connection-limit.md` |
| The instability is a property of H4D | wrong; it is a property of `verbs;ofi_rxm`. PSM3 validates 20/20 | `results/h4d-psm3.md` |
| The streamed exchange improves stability | wrong; 7/20 against 9/20 at n=20. Its symmetric-window reduction is real | `docs/streamed-exchange.md` |
| `FI_VERBS_GID_IDX` and `FI_OFI_RXM_CQ_EQ_FAIRNESS` improve ISx64 | not demonstrated; four arms at n=20 were indistinguishable | — |

Two method notes that caused several of these:

**Five runs cannot separate 20% from 80%.** Three separate n=5 samples pointed the wrong
way. Every completion rate here is n=20 or is labelled provisional.

**`oshcc` embeds an RPATH that overrides `LD_LIBRARY_PATH`.** Two SOS installs on one
machine will silently load the wrong one. Use `LD_PRELOAD` and confirm with `ldd` before
trusting any A/B between builds.
