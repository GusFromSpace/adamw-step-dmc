# adamw-step-dmc

A [demoniC](https://github.com/GusFromSpace/demoniC) port of
[**PyTorch torch.optim.AdamW**](https://arxiv.org/abs/1711.05101 (Loshchilov & Hutter, 2019)).

## Files

- `adamw_step.dmc` — the demoniC port
- `adamw_step.py` — Python reference (the same algorithm, for verification)

## Verification

The port is checked out-of-band so the `.dmc` stays a pure translation. `verify/run.sh`
runs `adamw_step.py` and the `.dmc` on the same fixed inputs and confirms every
emitted number agrees (rtol 1e-5):

```
DMC=/path/to/dmc verify/run.sh
```

## License & attribution

**Unlicensed.** This is a clean-room reimplementation of a *published algorithm*, not a copy of any source code, so no upstream code license applies. For the original work, defer to the original author. ¯\_(ツ)_/¯  See [NOTICE](NOTICE).

Credit for the original goes to the PyTorch authors. See [NOTICE](NOTICE).
