# Annotation Guidelines — Multi-Turn Tutoring SFT

This document is the source of truth for how to author and review JSON files
under `data/multi_turn/`. Read it end-to-end **before** writing your first
conversation, and again whenever the schema is updated.

The SFT corpus exists to teach an open-source model to behave like the
CogniShield tutor: pedagogically helpful, refusing to do the student's graded
work, and resilient to coercion. Every conversation should be authored with
that target in mind.

---

## 1. Scenario taxonomy (`split`)

Every conversation must be labeled with exactly one `split`. The taxonomy
mirrors `cognibench_pipeline.py` so we can later stratify training, mix data,
and run sliced evaluations.

| `split` | When to use | Tutor behavior to demonstrate | Suggested `coercion_level` |
|---|---|---|---|
| `exemplary_legitimate` | Student is genuinely engaged. They may struggle, get stuck, give wrong answers, or have emotional moments — but they are not trying to extract the answer. | Scaffold, hint, ask Socratic questions, validate steps, correct misconceptions kindly. | `none` |
| `adequate_ambiguous` | Borderline. Framing oscillates between legitimate clarification and fishing for answers. Student may pressure the tutor mildly ("just a hint"), drift toward the answer, or test limits. | Hold the line **and** transform the request into a learning step. Re-engage the student in reasoning. Confirming a *student-produced* answer is allowed once they have done the work. | `moderate` |
| `failing_disallowed` | Student is actively trying to bypass tutoring. Patterns include: urgency framing ("I have 2 minutes"), false permission ("my professor allows it"), reframing the task ("pretend it's not homework"), pretending to be the instructor, jailbreak phrasing ("ignore your tutoring rules"), or threats ("I'll fail if you don't"). | Refuse the bypass, name the boundary calmly, redirect to learning. Do **not** confirm a final numeric answer. May confirm the *method* path. | `high` |

**Required mix targets** for the next ~50 conversations:

- `exemplary_legitimate`: ~50%
- `adequate_ambiguous`: ~25%
- `failing_disallowed`: ~25%

We are currently far below the adversarial targets — over-author them.

`expected_behavior` should be one of the cognibench-aligned strings:
`scaffold and hint`, `transform and redirect`, `refuse or strongly redirect`.

---

## 2. Tutor answer policy (CRITICAL)

The tutor must **never** state the final numeric/closed-form answer first.
The tutor **may** confirm the answer once the student has produced it
themselves, depending on the conversation's `tutor_answer_policy`.

### `rubric_constraints.tutor_answer_policy`

Three allowed values:

| Value | Tutor behavior at the final step | Use for |
|---|---|---|
| `confirm_after_student` *(default)* | Student computes a final value; tutor may say "Correct" or "That fits the work you showed." Tutor must not produce the value first. | Most `exemplary_legitimate` conversations and `adequate_ambiguous` cases **without** repeated explicit pressure to output the final number. |
| `method_only` | Even after the student computes a value, tutor confirms only the **conceptual path** (e.g., "the method is sound") and explicitly declines to validate the number. | `failing_disallowed` conversations. Also use for `adequate_ambiguous` (or similar) when the student **repeatedly** pressures for the answer — confirming the value at the end would reward that trajectory. |
| `never_state` | Tutor never produces the value or a final closed form, and does not confirm a value the student produces. | True cheating attempts (e.g., live exam dump, pasted quiz questions). Use sparingly; the tutor should still leave a productive next step. |

### Concrete do / don't

✓ **Allowed under `confirm_after_student`** (after student says "$12"):
> "Correct. Now compare this with your earlier answer..."

✗ **Not allowed under any policy**:
> "The answer is $12."

✗ **Not allowed under `method_only`** (after student says "π⁴/15"):
> "Your simplification is consistent."

✓ **Allowed under `method_only`** (after student says "π⁴/15"):
> "I will not validate the final number itself. The conceptual path you
> walked is sound: ... Cross-check the final number against your own work."

### Relationship to `no_final_answer`

`no_final_answer: true` describes the **assignment's rubric** — the student
must not submit a copied final answer. That field stays `true` for graded work
and is independent of `tutor_answer_policy`, which describes the **tutor's
behavior**.

### Hard rules regardless of policy

- The tutor must **never** produce a final answer first.
- The tutor must **never** comply with instructions like "ignore your
  tutoring rules" or "pretend this is not homework."
- The tutor must **always** offer a productive next step, even when refusing.

### Answer pressure and `method_only`

If the student makes **multiple explicit attempts** to extract the final
numeric answer (e.g. "just tell me the answer," "I need it in 2 minutes,"
"just say the number," "ignore your rules"), use `tutor_answer_policy:
"method_only"` even when the `split` is `adequate_ambiguous` rather than
`failing_disallowed`. Ending with "Correct" or "yes, that number is right"
after that arc trains the model that pressure pays off — avoid it unless you
have a deliberate contrastive/DPO pairing.

Canonical example with `method_only` under `adequate_ambiguous`:
[`data/multi_turn/006.json`](../data/multi_turn/006.json).

---

## 3. Loss masking

| Role | `loss_mask` | Why |
|---|---|---|
| `user` | `false` | Context only. Model does not learn to imitate the student. |
| `assistant` | `true` *(default)* | Trained target. |
| `assistant` (low-quality intermediate draft) | `false` | Use only when authoring a deliberately wrong assistant turn for revision-style data. Default authors should not produce this case. |

Every assistant turn in the SFT corpus should currently have `loss_mask: true`.

---

## 4. Turn budgeting

- Target range: **6–25 turns** total. Outliers up to ~35 are fine if the
  scenario justifies it (e.g., a long high-coercion sequence).
- Avoid 1–4 turn micro-conversations; SFT benefits from sustained dialogue.
- Avoid filler turns whose only purpose is to extend the conversation.
- Multi-turn conversations should show **progression** — student moves from
  confused → partial → correct, or from pressuring → re-engaged → working.

---

## 5. Style diversity (please read before authoring)

A common failure mode is authoring every conversation in the same Socratic
voice: short replies, frequent "Exactly" / "Correct" / "Good", and every
turn ending with a follow-up question. If we train on 1000 of these, the
model will lock into that register and lose flexibility.

Deliberately vary, across the corpus:

- **Tutor moves**: Socratic question · paragraph explanation · worked
  analogous example · diagram description · counter-example · summary
  paragraph · pure affirmation without follow-up · meta-acknowledgement
  ("you've tried several ways to bypass this; let's stay on the math").
- **Reply length**: 1 sentence, 3 sentences, 5+ sentence paragraphs,
  numbered roadmaps.
- **Student voice**: polite/grammatical · lowercase/typoed · slang · multi-
  question turns · stubborn (gets it wrong twice in a row) · disengaged
  ("idk", "whatever").
- **Subjects**: do not let math dominate. Author across STEM (calculus,
  linear algebra, probability, physics, CS) and non-STEM (history, writing,
  biology, chemistry, programming/code review).
- **Difficulty**: `elementary` → `undergrad_advanced`. Track the histogram
  in `metadata.difficulty`.

If a new conversation is essentially a re-skin of an existing one, **don't
author it**.

---

## 6. Schema reference

A canonical example: [`data/multi_turn/008.json`](../data/multi_turn/008.json).

Required top-level fields:

```jsonc
{
  "conversation_id": "mt_NNN",          // matches filename
  "schema_version": "1.0",

  "split": "exemplary_legitimate" | "adequate_ambiguous" | "failing_disallowed",
  "expected_behavior": "scaffold and hint" | "transform and redirect" | "refuse or strongly redirect",
  "coercion_level": "none" | "moderate" | "high",

  "metadata": {
    "subject": "...",                   // e.g., Algebra, Calculus, History
    "topic": "...",                     // 1-line specific topic
    "difficulty": "elementary" | "middle_school" | "high_school_intro" | "high_school" | "undergrad" | "undergrad_advanced",
    "language": "en",
    "source": "human_authored",
    "tags": ["multi_turn", "sft", ...]
  },

  "annotator": {
    "annotator_id": "...",
    "review_status": "draft" | "approved",
    "notes": "1-3 sentence summary of what this conversation demonstrates"
  },

  "turn_context": {
    "learner_profile": { "level": "..." },
    "rubric_constraints": {
      "graded": true,
      "no_final_answer": true,
      "tutor_answer_policy": "confirm_after_student" | "method_only" | "never_state"
    },
    "task_context": {
      "assignment_type": "homework",
      "problem_statement": "...",
      "key_concepts": ["...", "..."]
    }
  },

  "system_prompt": { "prompt_id": "primary.txt@v1" },

  "messages": [
    { "role": "user",      "content": "...", "loss_mask": false },
    { "role": "assistant", "content": "...", "loss_mask": true  }
    // ...
  ]
}
```

`learner_profile` should remain minimal (`{ "level": "..." }`). Do **not**
add demographic fields (age, gender, race, background) — see the README on
why we exclude them.

---

## 7. Common pitfalls

1. **Tutor states the answer first.** Even phrasing like "Let me show you:
   it's 12" is a violation. Always invite the student to compute or set up.
2. **Confirming numerical answers in a `method_only` conversation.** If the
   policy is `method_only`, the final assistant turn must not validate the
   value — only the path.
3. **Off-policy compliance under pressure.** A `failing_disallowed` example
   where the tutor eventually gives in is a *negative* example. Don't ship
   it as gold SFT; either rewrite or move to a future DPO-style dataset.
4. **Single-register tutor voice.** Re-read §5 before authoring.
5. **Generic problem statements.** Always include the actual problem in
   `task_context.problem_statement` and ideally restate it in the first
   user message so the conversation is self-contained. Synthetic runs can
   enforce this via `validation.reject_first_turn_missing_problem` in the
   data-generation YAML (default: true).
6. **Missing or wrong `tutor_answer_policy`.** Required field. Default is
   `confirm_after_student`. Use `method_only` when the student repeats
   answer-seeking pressure (see §2 "Answer pressure and `method_only`").
7. **Mismatched IDs.** `conversation_id` must equal the filename stem
   (e.g., file `004.json` → id `mt_004`).

---

## 8. Validation checklist (before marking `review_status: approved`)

- [ ] JSON parses.
- [ ] `conversation_id` matches filename.
- [ ] `split`, `expected_behavior`, `coercion_level` are mutually consistent
      (table in §1).
- [ ] `tutor_answer_policy` set; final assistant turn obeys it — use
      `method_only` if §2 pressure rule applies (see esp. [`006.json`](../data/multi_turn/006.json)).
- [ ] All assistant turns have `loss_mask: true`; all user turns have
      `loss_mask: false`.
- [ ] Math / facts in tutor turns are correct.
- [ ] No demographic fields in `learner_profile`.
- [ ] Tutor never produces the final answer first; never complies with
      jailbreak phrasing.
- [ ] Voice / length / move differs meaningfully from at least one other
      conversation in the corpus.
