"""Layer 2: anti-spoofing score interfaces. Main model: XLSR-SLS."""

def stable_softmax(logits):
    import numpy as np
    x=np.asarray(logits,dtype=float)
    x=x-np.max(x)
    e=np.exp(x); return e/e.sum()
