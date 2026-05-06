# LLM Agent Prompts

Two prompts drive the agenda-search baseline that produces the `community_papers` field in `data/queries_80_with_graph_search.json`:

| File | Role |
|---|---|
| `search_planner.txt` | **Search planner** — converts a high-level research agenda into a boolean keyword query suitable for the paper-search API. Outputs the `condition` field (filter tree of OR / AND keyword groups). |
| `parallel_evaluator.txt` | **Parallel evaluator** — for a candidate paper batch returned by the keyword search, decides which papers are on-topic for the agenda. Produces the relevance label that gates entry to `community_papers`. |

Both prompts are plain-text Jinja templates and were executed against `gemini-2.5-pro` on Vertex AI. Any sufficiently strong instruction-following model should work; rerunning either prompt regenerates the corresponding stage of the pipeline.
