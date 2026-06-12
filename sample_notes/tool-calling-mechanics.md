# Tool-Calling Mechanics

In **tool calling**, the model is given tool schemas and emits a structured call
(name + arguments); the system executes it and returns the result as an
observation, which the model reads before continuing. Forcing a tool call (rather
than free text) guarantees a machine-parseable, schema-valid output.
