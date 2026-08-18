# tokin
Make any agent harness token-native.

<p align="center">
  <img src="assets/architecture.svg" alt="The harness sends messages to tokin and gets back text and tool calls. tokin sends input_ids to the inference server and gets back output_ids and logprobs." width="100%">
</p>

The harness keeps speaking text over a standard chat-completions API. The inference
server only ever receives token ids. tokin owns the chat template in between, so the
ids a model generated on one turn are the ids the next turn's prompt carries — no
detokenize/retokenize round trip, no silent drift.
