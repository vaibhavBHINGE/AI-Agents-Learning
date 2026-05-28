# ✨ Prompt Engineering Techniques

Here we will see prompt techniques, how they work, and why these techniques are required.

---
git 
# What is Prompt?

A **Prompt** is a message/input or instruction given to the AI to generate a response.

There are several techniques to create prompts. Some of them are mentioned below as they are widely used.

---

# 1. Zero-shot Prompting

In simple terms, a **shot** is an example.

A prompt that doesn't contain any example is called a **Zero-shot Prompt**.
Zero-shot directly instructs the AI without any example.

## Example

### Prompt

```text
What is the capital of Papua New Guinea?
```

### Answer

```text
Port Moresby
```

---

# 2. Few-shot Prompting

This prompt uses one or more than one example to instruct the LLM.

If Zero-shot or One-shot prompting is not able to provide the expected output, then we use this Few-shot technique.

## Example

### Prompt

```text
India: New Delhi :: China: Beijing :: Japan: Tokyo :: USA: ?
```

### Answer

```text
India: New Delhi :: China: Beijing :: Japan: Tokyo :: USA: Washington, D.C.
```

---

# 3. Chain of Thoughts (CoT)

Chain of Thought is a technique where the LLM follows step-by-step reasoning to reach a final answer.

This breaks complex problems into manageable steps, allowing the AI to "think" through the logic.

## Example

### Prompt

```text
I have 3 baskets. Each basket has 5 apples.
I give 2 apples to my neighbor and eat 1 myself.
How many apples do I have left?
```

### Reasoning

```text
First, calculate the total apples:
3 baskets × 5 apples = 15 apples.

Next, subtract the apples given away:
15 - 2 = 13 apples.

Finally, subtract the apple eaten:
13 - 1 = 12 apples.
```

### Answer

```text
12
```
