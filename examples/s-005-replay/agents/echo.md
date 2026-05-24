---
agentId: echo
version: 0.0.1
fsm:
  states:
    - name: react
      type: llm
      instructions: respond with the input verbatim
      tools: []
model:
  provider: stub
  model: stub
  adapter: openai-compatible
---
You echo the user's input verbatim.
