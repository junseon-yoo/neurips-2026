# LLM Agent Prompts

Two LLM agents are used in this work:

- **search-strategist** — converts a high-level research interest into a boolean keyword query for a paper-search API
- **research-analyst** — given a candidate paper set + a research interest, decides which papers belong to the same research agenda (used to seed the 80-query benchmark)

Place the prompt templates here:

```
prompts/
├── search_strategist.txt
└── research_analyst.txt
```

Both prompts are plain-text Jinja-style templates. They are executed against
Gemini-2.5-Pro (`gemini-2.5-pro` on Vertex AI) — you can substitute any
sufficiently strong instruction-following model.
