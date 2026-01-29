# Quick Reference: Grid Size Limits

## The Golden Rule
**Maximum: nx + ny ≤ 24 qubits**

## Quick Lookup Table

| Want to Run | Use This | Total Q | Status |
|-------------|----------|---------|--------|
| Quick test | 3×3 | 6 | ✅ Instant |
| Light sim | 5×5 | 10 | ✅ Fast |
| Normal sim | 8×8 | 16 | ✅ Good |
| Detailed sim | 10×10 | 20 | ⚠️ Slower |
| Maximum res | 12×12 | 24 | ⚠️ Slow |

## What NOT to Do
- ❌ 50×50 (100 qubits) - Will crash
- ❌ 200×200 (400 qubits) - Will crash
- ❌ Any grid where nx+ny > 24 - Will crash

## Why It Fails
Quantum state vectors need `2^(nx+ny)` memory slots.

```
8×8 grid:   2^16 = 65,536 slots ✅
50×50 grid: 2^100 = way too many ❌
```

## If You Get an Error
```
❌ Grid too large! Total qubits: 25 exceeds max 24
💡 Reduce grid sizes so nx + ny ≤ 24
   Example: nx=12, ny=12
```

**Solution:** Adjust nx and/or ny so they add up to ≤24.

## Recommended Setups

### For Testing
```python
nx = 3
ny = 3
# Total: 6 qubits → runs instantly
```

### For Quality Results
```python
nx = 8
ny = 8
# Total: 16 qubits → good balance of speed and resolution
```

### For Maximum Detail
```python
nx = 12
ny = 12
# Total: 24 qubits → maximum allowed, slowest
```

### For Wide Domains
```python
nx = 18
ny = 6
# Total: 24 qubits → maximum allowed, rectangular
```

---

**Updated:** Phase 5 - Memory Overflow Prevention
**Status:** ✅ Ready to use
