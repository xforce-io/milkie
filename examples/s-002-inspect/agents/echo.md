---
agentId: echo
version: 0.0.1
fsm:
  states:
    - name: react
      type: llm
      instructions: greet the user briefly
      tools: []
model:
  provider: stub
  model: stub
  adapter: openai-compatible
---
You are a friendly greeter.
