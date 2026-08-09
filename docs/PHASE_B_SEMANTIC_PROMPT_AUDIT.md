# Phase-B semantic-control prompt audit

## Purpose

The named/functional and obfuscated/opaque prompts must define the same
scientific task without making the semantic control easier by disclosing a more
detailed causal graph. This audit covers the multi-target or multi-input tasks:
Dalla Man T2--T4 and CSTR, plus the tier-specific Alien Device obligations. T1
is excluded because its single target and single external event leave no
ambiguous endpoint mapping to disclose.

## Equivalence rule

Each named/obfuscated pair must preserve:

1. the same observable prediction obligations;
2. the same number and scope of required mechanisms; and
3. the same level of structural specificity.

Obfuscation may replace semantic names with neutral symbols and role-level
descriptions. It must not additionally reveal exact right-hand-side placement,
direct-versus-mediated effects, signs, or a complete input-to-state graph.

## Audit results

| Task | Required mechanisms | Preserved in obfuscated prompt | Deliberately withheld |
|---|---:|---|---|
| T2 | 2 | Delayed input response; delayed regulator-dependent removal | Exact removal target/balance placement and the endpoint of the second input |
| T3 | 3 | Delayed input response; peripheral regulatory removal; distinct regulated source | Exact balance placement and direct mappings of the second and third inputs |
| T4 | 1 composite portrait | Input response, regulatory removal, internal source, exchange, and secondary-target generation | Full flux graph and direct input-to-target assignments |
| CSTR | 1 composite balance | External transport, state-dependent internal source, and coupled exchange | Feed/input endpoint assignments and the explicit reaction/exchange graph |
| Alien Device, easy | 1 | Input-driven memory and its causal contribution to the target | Device semantics and telemetry meanings |
| Alien Device, hard | 1 composite pathway | Input memory, persistent coupling, nonlinear feedback, and target generation | Device semantics and private state identities |

The mechanism counts match the named prompts in both easy and hard tiers and,
for Dalla Man, in both canonical and perturbed dynamics. The Alien Device
functional/opaque pairs also preserve the distinct easy and hard mechanism
burdens. Public channel lists still declare every available numeric symbol, but
descriptions no longer encode endpoint-level causal assignments.

## Regression checks

Automated tests enforce matching mechanism counts and reject the previously
leaking phrases, including exact entry into the primary-target balance, direct
input-to-target contribution, and explicit CSTR input endpoint mappings. These
checks address prompt parity only; manual scientific review remains required
before the prompts are finalized.
