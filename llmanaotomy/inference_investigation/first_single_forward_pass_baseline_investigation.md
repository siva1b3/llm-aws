# First intended investigation (not yet executed)

Establish the minimal, real black-box baseline that performs **one forward call**, selects one next-token ID, and decodes it. Once executed, record only what was observed: invocation, the input, model-output/logit shape, selected token ID, and decoded token. Then stop for learner confirmation before peeling the next layer. Do not mark this step complete based on its description.
