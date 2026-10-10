# Learning notes

This project exists to **learn fine-tuning end to end**. These pages are the story and the study notes, written
so that future-you (or anyone new) can understand the code *and* why it is the way it is.

## How to read these pages

| Page | What it is | Read it when |
|---|---|---|
| [00-story-so-far.md](00-story-so-far.md) | The project as a story: goal, what was built in order, the decisions and why, the numbers | You come back after a break, or want the big picture |
| [glossary.md](glossary.md) | Every term in plain words, with the step where it first appears | You meet a word you do not remember |
| `step-3-N-*.md` | One page per DistilBERT step: what we did, code map, how to re-run, what we saw, **the questions and full answers** | You want to understand or re-do a step |
| [../math/](../math/README.md) | The maths, with worked numbers checked against code | You want the formulas |
| [../debugging.md](../debugging.md) | Logs, request ids, the Debug tab | Something looks wrong |
| [../prds/](../prds/00-overview.md) | The plan: what each step is meant to deliver | You want to know what comes next |

## The rule we follow

After **every step**, a page is added here with the three "check your understanding" questions *and their answers*, plus
any other question you asked along the way and anything surprising we found. If a question comes up in a conversation, it
goes on that step's page. Nothing important should live only in a chat.

## Pages

| Step | Page |
|---|---|
| Whole project so far | [00-story-so-far.md](00-story-so-far.md) |
| TF-IDF baseline (PRD 3A) | [tfidf-baseline.md](tfidf-baseline.md) |
| 3.1 GPU environment | [step-3-1-environment.md](step-3-1-environment.md) |
| 3.2 Tokenization | [step-3-2-tokenizer.md](step-3-2-tokenizer.md) |
