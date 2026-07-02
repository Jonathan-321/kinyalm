# Assignment 1 Start Plan

Assignment 1 is where the model stops being a black box. Treat each test group
as a concept checkpoint.

## Collaboration Boundary

The CS336 assignment repo asks AI tools to act like teaching assistants, not
solution generators. For this workspace, that means:

- I can explain concepts.
- I can help you read tests and handout requirements.
- I can suggest sanity checks and debugging questions.
- I can review code you wrote.
- I should not write the assignment implementation for you.
- I should not complete TODOs or provide pasteable solutions.

That boundary is useful. It keeps the hard part where the learning happens.

## First Pass Order

1. **Linear, embedding, SiLU, softmax, cross-entropy**

   Goal: get comfortable with tensor shapes and logits-to-loss flow.

2. **RMSNorm and SwiGLU**

   Goal: understand the non-attention parts of a transformer block.

3. **Scaled dot-product attention**

   Goal: understand query/key/value and causal masking.

4. **Multi-head attention**

   Goal: understand how many heads are packed into batched matrix operations.

5. **RoPE**

   Goal: understand how token position enters attention.

6. **Transformer block**

   Goal: connect normalization, attention, residual paths, and feed-forward
   layers.

7. **Transformer LM**

   Goal: follow token IDs through embeddings, blocks, final norm, and output
   logits.

8. **Optimizer, schedule, clipping, checkpointing**

   Goal: understand what makes training stable and restartable.

9. **BPE tokenizer**

   Goal: understand how raw text becomes token IDs.

10. **Minimal training run**

    Goal: train a small model and inspect loss movement.

## For Each Component

Use the same rhythm:

1. Read the handout section.
2. Inspect the related tests.
3. Write down expected input and output shapes.
4. Make a tiny hand-checkable example.
5. Implement your version.
6. Run only the related tests.
7. Add a short explanation to your notes.

## First Study Question

Before implementing anything, answer this in your own words:

```text
For one batch of token IDs, what tensors are created between input tokens and
the final logits?
```

If that sentence feels fuzzy, start with embeddings, linear layers, softmax, and
cross-entropy before touching attention.
