# Hybrid LLM Routing Use Case

## Goal

Show a minimal but useful architecture where:

- a large LLM understands natural language,
- small QLANG models make cheap, repeatable routing decisions,
- QLANG carries the structured state between components.

The first use case is an **agent gateway for incoming requests**.

## Problem

A general-purpose LLM can classify and route requests, but doing that with
free-form text on every hop is expensive and unstable. The same request may be
phrased differently, intermediate decisions are hard to validate, and it is
unclear which part of the system is responsible for which decision.

## Proposed Flow

1. The user sends a natural-language request.
2. A large LLM acts as the planner and converts the request into structured
   routing scores and risk scores.
3. QLANG encodes those scores as typed tensors in graph messages.
4. Small QLANG specialists execute lightweight decision graphs.
5. The planner receives the structured outputs and decides the next action.

## Why This Is A Good First MVP

- The small-model task is narrow and easy to label.
- The outputs are discrete and easy to evaluate.
- The benefit of binary, typed exchange is visible immediately.
- The system still leaves the large LLM where it is strongest: language
  understanding and final response synthesis.

## What Should Be Trained Later

The small QLANG models should eventually be trained on:

- request intent classification
- route prediction
- escalation / risk prediction

For the first demo, those specialist models can be represented as deterministic
QLANG graphs operating on planner-produced score tensors.

## Success Criteria

The MVP is successful if it can:

- accept a user request,
- produce typed routing state,
- send it via QLANG messages,
- resolve a route and risk level through small specialist graphs,
- and return a final orchestration decision without requiring text-only hops.
