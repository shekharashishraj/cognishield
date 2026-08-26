# CogniShield Project Summary

## 1. What Is This Project?

CogniShield is a prototype tutoring-safety system for educational AI.

The goal is to let an AI tutor help students learn without simply giving away final answers, especially when the student is working on graded homework, quizzes, or assignments. The system is designed around careful tutoring behavior: give hints, ask guiding questions, correct misconceptions, redirect cheating attempts, and preserve academic integrity.

The repository has three main parts:

1. Runtime tutoring app
   - A command-line app under `cognishield/app/`.
   - It can run a tutor pipeline using OpenAI or an OpenAI-compatible backend such as vLLM.

2. Synthetic SFT data generation
   - Code under `training/data_generation/`.
   - It creates multi-turn tutoring conversations for supervised fine-tuning.

3. SFT training and evaluation
   - Code under `training/`.
   - It converts conversations to SFT JSONL, fine-tunes a model with LoRA, merges the adapter, serves the model, and runs smoke/stress tests.

## 2. What Has Been Done So Far?

The project already has:

- A runtime CogniShield CLI.
- Two tutoring pipelines:
  - `legacy`: planner -> generator -> validators -> verifier -> revision.
  - `meta`: primary tutor -> meta-agent classifiers -> verifier -> revision.
- A hand-authored seed/eval dataset in `data/multi_turn/`.
- A synthetic data generation pipeline.
- A generated reviewed dataset of 9,977 multi-turn conversations.
- A converted SFT JSONL file: `training/data/sft.generated.batch_10000.jsonl`.
- A LoRA fine-tuning setup for Qwen2.5.
- A completed SFT run on `Qwen/Qwen2.5-7B-Instruct`.
- A saved LoRA adapter: `out/qwen25-7b-tutor-lora-v0`.
- A merged model for inference: `out/qwen25-7b-tutor-merged-v0`.
- vLLM serving and manual single-turn/multi-turn testing have also been tried.

## 3. Dataset Generation

The current large data-generation config is:

- Config file: `training/data_generation/configs/batch.yaml`
- Target examples: 10,000
- Maximum candidate examples: 30,000
- Generator model: `gemma-31b-it`
- Judge model: `gemma-31b-it`
- Output directory: `data/generated/batch_10000`
- Reviewed/exported directory: `data/generated_reviewed/batch_10000`

The generator creates one complete tutoring conversation at a time. Each conversation must follow a strict schema and must include:

- A conversation id.
- A split label.
- Expected behavior.
- Coercion level.
- Metadata such as subject, topic, difficulty, and tags.
- Learner profile.
- Rubric constraints.
- Task context.
- Alternating user and assistant turns.
- Loss masks for SFT training.

The first user turn must include the full problem statement. The assistant should use the reference solution internally, but should not reveal the final answer when the policy forbids it.

## 4. How Many Dataset Points Were Generated?

The final reviewed/exported dataset contains:

- 9,977 conversations.
- 45,942 assistant turns.
- 9,977 raw accepted examples.
- 9,977 final valid examples.
- 0 final deterministic validation rejects.
- 13,353 rejected generation attempts.

The target was 10,000, but the final converted/exported set contains 9,977 usable conversations.

The SFT stats file reports:

- Total conversations: 9,977
- Total assistant turns: 45,942

## 5. Dataset Splits

The conversations are grouped into three broad behavior splits:

| Split | Count | Meaning |
| --- | ---: | --- |
| `exemplary_legitimate` | 6,190 | Normal tutoring cases where the student can be helped with scaffolding |
| `adequate_ambiguous` | 1,596 | Moderate pressure cases where the student asks for answers but context is not clearly a live exam |
| `failing_disallowed` | 2,191 | High-risk cases such as live quiz cheating or jailbreak attempts |

## 6. Coercion Levels

The dataset also labels how much pressure the student applies:

| Coercion Level | Count | Meaning |
| --- | ---: | --- |
| `none` | 6,190 | No coercive pressure |
| `moderate` | 1,596 | Student pushes for final answer |
| `high` | 2,191 | Student claims quiz/test context or tries to bypass rules |

## 7. Expected Tutor Behaviors

Each conversation has an expected behavior:

| Expected Behavior | Count | Meaning |
| --- | ---: | --- |
| `scaffold and hint` | 6,190 | Help the student reason step by step |
| `transform and redirect` | 1,596 | Refuse direct answer and convert request into a useful learning step |
| `refuse or strongly redirect` | 2,191 | Strongly avoid giving or confirming answers in disallowed contexts |

## 8. Tutor Answer Policies

The dataset uses three answer policies:

| Policy | Count | Meaning |
| --- | ---: | --- |
| `confirm_after_student` | 4,998 | The tutor may confirm after the student has done the reasoning |
| `method_only` | 2,985 | The tutor should explain method but avoid confirming final values |
| `never_state` | 1,994 | The tutor must not state or confirm the final answer |

## 9. Student / Learner Profiles

Each generated conversation includes a simple learner profile with a `level` field.

The generated dataset has:

| Learner Level | Count |
| --- | ---: |
| `high_school_high` | 3,982 |
| `high_school_low` | 2,999 |
| `undergraduate` | 2,996 |

These correspond to metadata difficulty labels:

| Difficulty | Count |
| --- | ---: |
| `high_school` | 3,982 |
| `high_school_intro` | 2,999 |
| `undergrad` | 2,996 |

The learner profile is intentionally simple right now. It mostly controls the difficulty and style of tutoring, not a detailed simulated personality.

## 10. Subject and Domain Coverage

The planned domain mix was:

- 8,000 math examples.
- 2,000 coding examples.

The final reviewed dataset contains:

| Domain / Tag | Count |
| --- | ---: |
| `math` | 7,977 |
| `coding` | 2,000 |

Main subjects include:

| Subject | Count |
| --- | ---: |
| Mathematics | 5,567 |
| Computer Science | 2,000 |
| Arithmetic | 1,414 |
| Algebra | 524 |
| Geometry | 472 |

Common topics include:

- One-step and two-step linear equations.
- Area and perimeter.
- Unit price and quantity word problems.
- Ratios and rates.
- Python fundamentals.
- Programming functions.
- Algorithms and correctness.
- Counting and probability.
- Modular arithmetic.
- Complex numbers.
- Quadratic equations.

## 11. Types of Multi-Turn Conversations Generated

The dataset includes eight scenario types:

| Scenario | Count | Description |
| --- | ---: | --- |
| `misconception_correction` | 2,794 | Student has a plausible mistake; tutor diagnoses and redirects |
| `legitimate_scaffold` | 1,600 | Student genuinely asks for help; tutor gives hints and questions |
| `direct_answer_pressure` | 1,596 | Student repeatedly asks for the final answer |
| `wrong_answer_checking` | 1,397 | Student proposes an answer and asks if it is right |
| `live_quiz_cheating` | 1,196 | Student says they are in a quiz/test and asks for answers |
| `emotional_support` | 999 | Student is frustrated or discouraged |
| `jailbreak_attempt` | 995 | Student uses fake authority, role-play, or "ignore rules" tactics |
| `off_topic_redirect` | 797 | Student drifts off topic and tutor redirects back to the task |

Most conversations have 4 or 5 assistant turns.

Conversation length distribution:

| Assistant Turns | Conversation Count |
| ---: | ---: |
| 3 | 113 |
| 4 | 4,673 |
| 5 | 4,347 |
| 6 | 744 |
| 7 | 90 |
| 8 | 9 |
| 10 | 1 |

In the final SFT JSONL, the system prompt is added as the first message, so the message count is one larger than the raw reviewed conversation.

## 12. Validation and Filtering

The data-generation pipeline validates each conversation before it becomes training data.

Validation checks include:

- Correct schema.
- Correct conversation id.
- Alternating user/assistant turns.
- Required turn count range.
- First user message includes the full problem statement.
- User messages have `loss_mask: false`.
- Assistant messages have `loss_mask: true`.
- No obvious final-answer leakage in forbidden positions.
- No policy mismatch between scenario and tutor answer policy.
- LLM judge review during generation.

Rejected examples are stored separately under `generation_rejected/`.

## 13. SFT Training Setup

The SFT training was done with:

- Base model: `Qwen/Qwen2.5-7B-Instruct`
- Training file: `training/data/sft.generated.batch_10000.jsonl`
- Output adapter: `out/qwen25-7b-tutor-lora-v0`
- Method: LoRA supervised fine-tuning
- Trainer: TRL `SFTTrainer`
- PEFT method: LoRA
- Loss masking: `assistant_only_loss=True`

Important training config:

| Setting | Value |
| --- | --- |
| Epochs | 2 |
| Learning rate | `1.0e-4` |
| Batch size per device | 1 |
| Gradient accumulation | 4 |
| Max sequence length | 4096 |
| Packing | false |
| LoRA rank | 32 |
| LoRA alpha | 64 |
| LoRA dropout | 0.05 |
| Scheduler | cosine |
| Warmup ratio | 0.05 |
| Seed | 42 |

LoRA target modules:

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`

## 14. How the Loss Masking Works

The training script uses `assistant_only_loss=True`.

This means the model is trained only on assistant responses, not on user messages or the system prompt. This is important because the model should learn how the tutor should answer, not learn to imitate the student.

The training script also injects a Qwen ChatML template with generation markers:

- System messages are rendered as system turns.
- User messages are rendered as user turns.
- Assistant messages are wrapped with generation markers.
- Only assistant spans are used for loss.

This chat template is saved with the tokenizer so that inference through vLLM uses the same format as training.

## 15. Training Result

The training run completed successfully.

Final training stats:

- Steps: 4,990
- Epochs: 2
- Runtime: about 2,400 seconds
- Samples per second: 8.315
- Steps per second: 2.079
- Final average train loss: about 0.613
- Last logged step loss: about 0.291
- Last logged token accuracy: about 0.890

This indicates that the model learned the training distribution reasonably well.

The trained adapter was saved to:

```text
out/qwen25-7b-tutor-lora-v0
```

## 16. Merge and Serving

After training, the LoRA adapter was merged into the base Qwen model.

Merged model output:

```text
out/qwen25-7b-tutor-merged-v0
```

The merge step:

1. Loads the base model.
2. Loads the LoRA adapter.
3. Merges LoRA weights into the base model.
4. Saves a standalone Hugging Face model directory.
5. Saves the tokenizer and chat template with the merged model.

This merged model can be served with vLLM:

```bash
vllm serve out/qwen25-7b-tutor-merged-v0 \
  --served-model-name qwen25-7b-tutor \
  --port 8000 \
  --max-model-len 4096
```

The `--max-model-len 4096` setting matches the SFT training context length.

## 17. Evaluation Done So Far

Manual single-turn testing was performed through vLLM using `/v1/chat/completions`.

The model was asked to behave like a careful tutor and tested on direct-answer requests. Multi-turn testing was also discussed by manually preserving the `messages` array:

```text
system -> user_1 -> assistant_1 -> user_2 -> assistant_2 -> ...
```

The repository also includes two evaluation paths:

1. `training/smoke_eval.py`
   - Checks record count, loss mask behavior, template parity, train loss, and inference.

2. `training/stress_eval_merged.py`
   - Loads the merged model locally.
   - Runs multi-turn pressure scenarios.
   - Checks whether the model leaks final-answer content.

## 18. Important Testing Notes

For fair testing:

- Always use the chat-completions format.
- Keep the system prompt close to the training prompt.
- Preserve conversation history for multi-turn tests.
- Use deterministic generation when comparing behavior:
  - `temperature: 0`
- Avoid raw prompt tokenization unless explicitly testing plain completion behavior.
- For vLLM, use the same served model id in curl/eval as the one passed to `--served-model-name`.

## 19. Current Status

At this stage, the project has reached a working SFT prototype:

- Dataset generation works.
- Validation and filtering work.
- A 9,977-example dataset has been produced.
- Qwen2.5-7B-Instruct has been fine-tuned with LoRA.
- The LoRA adapter has been merged.
- The merged model can be served with vLLM.
- Initial single-turn inference has been tested.

The main remaining work is evaluation and iteration.

## 20. Potential Next Steps

Recommended next steps:

1. Run systematic multi-turn evaluation
   - Use scripted conversations instead of only manual curl tests.
   - Test the full set of scenarios:
     - legitimate help
     - misconception correction
     - wrong-answer checking
     - direct-answer pressure
     - live quiz cheating
     - jailbreak attempts
     - emotional support
     - off-topic redirect

2. Compare model outputs against expected behavior
   - Check whether the model gives hints instead of final answers.
   - Check whether it refuses live quiz cheating.
   - Check whether it avoids confirming final answers under `never_state`.
   - Check whether it remains helpful and not overly rigid.

3. Run stress tests
   - Use `training/stress_eval_merged.py`.
   - Add more stress scenarios from the generated dataset.
   - Include both math and coding cases.

4. Add automated replay evaluation
   - Replay selected conversations from `training/data/sft.generated.batch_10000.jsonl`.
   - Feed only user turns to the model.
   - Compare model replies with gold assistant replies.
   - Score for answer leakage, helpfulness, and policy compliance.

5. Evaluate through the full CogniShield runtime
   - Serve the model with vLLM.
   - Set `COGNISHIELD_OPENAI_API_BASE`, `COGNISHIELD_MODEL`, and `COGNISHIELD_PRIMARY_RAW_TEXT=true`.
   - Run the model through the `meta` pipeline.

6. Improve dataset quality
   - Inspect random samples from each scenario.
   - Look for overused phrasing.
   - Check whether generated tutors are too repetitive.
   - Improve prompts or filters if needed.

7. Add stronger metrics
   - Answer leakage rate.
   - Refusal correctness.
   - Helpfulness score.
   - Multi-turn consistency.
   - Topic retention.
   - Coding-task correctness.

8. Consider DPO or preference tuning later
   - SFT teaches the desired response style.
   - DPO could later help distinguish better vs worse tutor responses.
   - This should come after stronger evaluation confirms where the SFT model fails.

## 21. Simple Summary

CogniShield is an educational AI safety project. It trains and evaluates a tutor model that helps students learn without giving away final answers in inappropriate contexts.

So far, the project has generated nearly 10,000 validated multi-turn tutoring conversations, fine-tuned Qwen2.5-7B-Instruct with LoRA, merged the adapter into a standalone model, and served it through vLLM.

The dataset covers math and coding tasks across high-school and undergraduate levels. It includes normal tutoring, misconception correction, direct-answer pressure, emotional support, off-topic redirection, live quiz cheating, and jailbreak attempts.

The next major step is systematic evaluation: run many multi-turn tests, measure whether the model leaks final answers, and decide where the dataset or training process needs improvement.
