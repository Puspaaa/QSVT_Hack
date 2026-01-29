# ⚠️ Grid Size Limits & Validation

## The Problem

Quantum state vectors have exponential size: `2^(nx+ny)`

When you set nx=200, ny=200, that's `2^400` which is impossibly large!

```
nx=200, ny=200  →  2^400 ≈ 10^120  ← Larger than atoms in universe!
nx=50, ny=50    →  2^100 ≈ 10^30   ← Still way too large
nx=25, ny=25    →  2^50  ≈ 10^15   ← Still too large
```

---

## Solution: Qubit Limit

**Maximum allowed:** `nx + ny ≤ 24` qubits

This allows:
```
2^24 = 16,777,216 ≈ 16.7 million state vector elements
Manageable with ~125 MB RAM
```

---

## Practical Grid Examples

### Within Limits (✅ Works)

| nx | ny | Total Qubits | Grid Points | State Size | Status |
|----|----|----|----|----|-------|
| 3 | 3 | 6 | 9 | 2^6 = 64 | ✅ Fast |
| 5 | 5 | 10 | 25 | 2^10 = 1K | ✅ Fast |
| 8 | 8 | 16 | 64 | 2^16 = 65K | ✅ OK |
| 10 | 10 | 20 | 100 | 2^20 = 1M | ✅ Manageable |
| 12 | 12 | 24 | 144 | 2^24 = 16.7M | ✅ Max |

### Beyond Limits (❌ Crashes)

| nx | ny | Total Qubits | Grid Points | State Size | Status |
|----|----|----|----|----|-------|
| 13 | 13 | 26 | 169 | 2^26 = 67M | ❌ Too large |
| 20 | 20 | 40 | 400 | 2^40 ≈ 1T | ❌ Way too large |
| 50 | 50 | 100 | 2500 | 2^100 | ❌ Impossible |
| 200 | 200 | 400 | 40000 | 2^400 | ❌ Impossible |

---

## How the Validation Works

**In the Streamlit sidebar:**
```python
if n_total > 24:
    st.error(f"Grid too large! Total qubits: {n_total} exceeds max 24")
    st.stop()
```

**When you try to run simulation with large grid:**
```python
if n_qubits > max_qubits:
    st.error("Grid too large!")
    st.stop()
```

---

## Recommended Grid Sizes

### For Fast Interactive Testing
- **Default:** 3×3 (9 points, instant)
- **Larger:** 5×5 (25 points, fast)
- **Bigger:** 8×8 (64 points, few seconds)

### For More Detailed Simulations
- **High-res:** 10×10 (100 points, ~10 seconds)
- **Very high:** 12×12 (144 points, ~30 seconds)

### Maximum Useful Size
- **Absolute max:** nx + ny = 24
- **Example:** 12×12 (144 grid points)
- **Example:** 20×4 (80 grid points)
- **Example:** 18×6 (108 grid points)

---

## Why This Limit Exists

**Quantum computing constraint:**
- State vector has 2^(number of qubits) elements
- Even simulating classically needs to allocate all elements
- Memory limits practical simulation to ~24-25 qubits
- Real quantum computers can use more qubits but work differently

---

## Error Messages You Might See

### Message 1: Initial Condition Setup
```
❌ Grid too large! Total qubits: 26 exceeds max 24
💡 Reduce grid sizes so nx + ny ≤ 24
   Example: nx=12, ny=12 (144 grid points, 2^24 state)
```
**Fix:** Reduce nx and/or ny in the sidebar

### Message 2: During Simulation
```
❌ Grid too large! Total qubits: 30 exceeds max 24
💡 Reduce grid sizes so nx + ny ≤ 24
```
**Fix:** Same as above - total qubits must be ≤ 24

### Message 3: Computation Warning
```
⚠️ Large grid: 20 qubits = 2^20 = 1,048,576 state elements
   Computation will be slow.
```
**Not an error** - just a warning. You can still run it, but it will take longer.

---

## Current Implementation

**Sidebar Check (Lines 28-30):**
- Shows current qubit count
- Warns if > 18 qubits
- Stops if > 24 qubits

**Simulation Check (Lines 378-384):**
- Double-checks before allocating state
- Catches any edge cases

---

## Try These Examples

### Small (Quick)
```
nx = 3, ny = 3  → 6 qubits, 9 grid points
```

### Medium (Good Balance)
```
nx = 8, ny = 8  → 16 qubits, 64 grid points
```

### Large (but Works)
```
nx = 10, ny = 10  → 20 qubits, 100 grid points
```

### Maximum Allowed
```
nx = 12, ny = 12  → 24 qubits, 144 grid points
```

Do NOT try:
```
nx = 50, ny = 50  → 100 qubits, crashes!
nx = 200, ny = 200  → 400 qubits, impossible!
```

---

## Status

✅ **Validation implemented**
- Sidebar checks grid size before user can proceed
- Simulation checks again before allocating state
- Clear error messages explain the limit
- Suggestions provided for valid combinations
