import numpy as np
ActDigitLen = 100
def strActionToDigits(act_string):
    t = [ord(c) for c in act_string]
    d_len = len(t)
    assert d_len <= ActDigitLen, ("Action string is tooo long! Don't be wordy. Or you can increase ActDigitLen above.")
    pad = [-1 for _ in range(ActDigitLen-d_len)]
    return (t+pad)

def digitsToStrAction(digits):
    if all([a==0 for a in digits]): return 'ActionSet3::N/A;N/A'
    arr = [chr(d) for d in digits.astype(int) if d >= 0]
    return ''.join(arr)

"""
'ActionSet3::ChangeHeight;100'
"""