# MachineZero program format

The MVP uses fixed-width three-byte instructions written as hexadecimal text:

```text
OPCODE OPERAND_A OPERAND_B
```

Comments begin with `#`.

`test.mz` is a deterministic example for AlienCPU seed `2026`:

```bash
machinezero predict examples/programs/test.mz --seed 2026 --budget 20
```
