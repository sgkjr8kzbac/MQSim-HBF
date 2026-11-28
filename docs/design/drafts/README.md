# Drafts for phase 2B part 2

Untested drafts left by the endurance-model implementation when phase 2B was split: the segment-level event
simulator (`endurance_simulator_draft.py.txt`, design §4) and the seeded request stream
(`workload_stream_draft.py.txt`). They are not part of the package (renamed `.txt`). Start from them or discard
them when implementing milestone (d); see PLAN.md, task T-B1d.

Status: the request stream and the simulator are merged into hbf/workload.py and hbf/endurance.py (the stream
draft is deleted). The simulator draft is kept only for its untested `VllmTracker` (vllm_pool), the remaining
item in docs/design/phase2b-sim-plan.md, "Next steps".
