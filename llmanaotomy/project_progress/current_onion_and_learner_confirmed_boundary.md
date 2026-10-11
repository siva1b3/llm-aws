# Current onion / learning boundary

```text
Input text → [ CURRENT: entire LLM / one forward pass ] → next-token scores
                                                        ↓
                                              one token → decoded text
```

**Previously verified components:** None yet in this implementation.  
**Currently being investigated:** Establishing that the single-pass black box really runs with the chosen model.  
**Hidden remainder:** Every internal operation, to be exposed **one by one** after the baseline is verified and the learner confirms.
