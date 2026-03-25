# Swarm LLM
A system for running multiple stateless LLMs communicating via shared entropic KV store.

## Basic use
Copy `dot_env.example` to `.env` and add your provider credentials, model names and nicknames. Can also set various default parameters.


```
# Build docker image
./swarm.sh build

# Run a two agent system for 10 cycles, showing agent output
./swarm.sh run -a kimi:minimal kimi:minimal -c 10 -w
```

JSON event file placed in `./output`.

## Experiment data
`./experiment_data`
The JSON event logs that form the basis of the paper are here, with extracted memory write/append traces, both raw form for analysis, and in Markdown, with metadata as headings. Also present are LLM-performed narrative analyses.

## Utilities
Terminal UI viewer of event JSON: 
`./scripts/swarm_viewer.py <JSON file>`

Full context window fiction generator. Attempt to generate book length works. Fails with refusal or repetition: 
`./run_fiction.py --model kimi -v`




