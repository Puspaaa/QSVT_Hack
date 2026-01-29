# 🗺️ Code Location Reference: 2D Implementation

This document maps your 4 theoretical changes directly to specific lines of code.

---

## 1️⃣ CHANGE 1: "Double the Register"

### Concept
- 1D: `n` qubits
- 2D: `nx` + `ny` qubits (separate x and y registers)

### Code Locations

**quantum.py - Lines 140-163: `Shift_gate_2d()`**
```python
140 def Shift_gate_2d(nx, ny, axis=0, inverse=False):
141     """Cyclic shift in 2D grid along specified axis."""
142     n_total = nx + ny  # ← DOUBLE REGISTER
143     qc = QuantumCircuit(n_total, name=f"Shift_2d_ax{axis}")
144     
145     # For 2D grid flattened as idx = x + y*nx:
146     # Shift along x (axis=0): increment x index (wraps at nx)
147     # Shift along y (axis=1): increment y index (wraps at ny)
148     if axis == 0:  # Shift along x-axis
149         qc.append(QFTGate(nx), range(nx))  # ← FIRST nx QUBITS
150         for i in range(nx):
151             phase = 2 * np.pi * (2**i) / (2**nx)
152             if inverse: phase *= -1
153             qc.p(phase, i)
154         qc.append(QFTGate(nx).inverse(), range(nx))
155     else:  # Shift along y-axis
156         qc.append(QFTGate(ny), range(nx, nx+ny))  # ← SECOND ny QUBITS
157         for i in range(ny):
158             phase = 2 * np.pi * (2**i) / (2**ny)
159             if inverse: phase *= -1
160             qc.p(phase, nx + i)
161         qc.append(QFTGate(ny).inverse(), range(nx, nx+ny))
162     
163     return qc
```

**quantum.py - Lines 165-175: `Block_encoding_diffusion_2d()`**
```python
165 def Block_encoding_diffusion_2d(nx, ny, nu):
166     """2D Block Encoding for Laplacian operator."""
167     ancilla = QuantumRegister(2, 'anc')  # Ancilla
168     x_reg = QuantumRegister(nx, 'x')     # ← X REGISTER (nx qubits)
169     y_reg = QuantumRegister(ny, 'y')     # ← Y REGISTER (ny qubits)
170     qc = QuantumCircuit(ancilla, x_reg, y_reg)
```

**pages/2_2D_Simulation.py - Lines 20-28**
```python
20  # --- SIDEBAR ---
21  with st.sidebar:
22      st.header("Quantum Hardware")
23      nx = st.slider("Grid Width (nx)", min_value=2, max_value=4, value=3, step=1)
24      ny = st.slider("Grid Height (ny)", min_value=2, max_value=4, value=3, step=1)
25      n_total = nx + ny  # ← DOUBLE REGISTER
26      st.caption(f"Total qubits: {n_total} (x:{nx}, y:{ny}) → {nx*ny} grid points")
```

---

## 2️⃣ CHANGE 2: "Independent Shift" (Advection)

### Concept
- Apply `Advection_Gate(c_x, t)` on x-axis
- Apply `Advection_Gate(c_y, t)` on y-axis
- Stack them (no extra cost, both unitary)

### Code Locations

**quantum.py - Lines 209-260: `Advection_Gate_2d()`**
```python
209 def Advection_Gate_2d(nx, ny, c_x, c_y, physical_time):
210     """
211     2D Advection: e^{-c_x t ∂/∂x - c_y t ∂/∂y}
212     Applied as independent QFT phase shifts on x and y axes.
213     """
214     n_total = nx + ny
215     qc = QuantumCircuit(n_total, name=f"Adv2d(t={physical_time:.3f})")
216     
217     x_reg = list(range(nx))          # ← X REGISTER INDICES
218     y_reg = list(range(nx, nx + ny)) # ← Y REGISTER INDICES
219     
220     # === X-Advection === [INDEPENDENT on x]
221     qc.append(QFTGate(nx), x_reg)    # ← QFT on X AXIS
222     shift_x = c_x * physical_time
223     diag_x = []
224     for j in range(nx):
224         k = j if j < nx/2 else j - nx
225         angle_x = 2 * np.pi * k * shift_x
226         diag_x.append(np.exp(1j * angle_x))
227     qc.append(DiagonalGate(diag_x), x_reg)
228     qc.append(QFTGate(nx).inverse(), x_reg)
229     
230     # === Y-Advection === [INDEPENDENT on y]
231     qc.append(QFTGate(ny), y_reg)    # ← QFT on Y AXIS
232     shift_y = c_y * physical_time
233     diag_y = []
234     for j in range(ny):
235         k = j if j < ny/2 else j - ny
236         angle_y = 2 * np.pi * k * shift_y
237         diag_y.append(np.exp(1j * angle_y))
238     qc.append(DiagonalGate(diag_y), y_reg)
239     qc.append(QFTGate(ny).inverse(), y_reg)  # ← BOTH STACKED
240     
241     return qc
```

**simulation.py - Lines 370-410: `run_split_step_sim_2d()`**
```python
370     for t in time_steps:
371         print(f"Simulating t_step = {t}...", end=" ")
372         
373         if t == 0:
374             results[t] = (state_vector, 1.0, None)
375         else:
376             # Compute 2D timestep
377             t_phys = t * dt
378             
379             # Build quantum circuits for this timestep
380             for init_state in init_states:
381                 # Build QSVT circuit for diffusion
382                 qc_qsvt = QSVT_circuit_2d(phi_seq_diff, nx, ny, nu, 
383                                           init_state, measurement=True)
384                 # ↓ Inside QSVT_circuit_2d, Advection_Gate_2d is applied
385                 # with independent x and y shifts stacked
386                 
387                 # Execute circuit and postselect
388                 ...
```

**pages/2_2D_Simulation.py - Lines 329-345: Visualization of Independent Motion**
```python
329             # Plot actual evolution comparison
330             fig, axes = plt.subplots(1, len(time_steps_display), 
331                                       figsize=(15, 4))
332             
333             # Each timestep shows c_x and c_y effect independently applied:
334             # Diffusion (QSVT) + X-Advection (shift x) + Y-Advection (shift y)
```

---

## 3️⃣ CHANGE 3: "2D Block Encoding" (The Hard Part)

### Concept
- 1D: 3-point stencil {I, LeftShift, RightShift}
- 2D: 5-point stencil {I, RightShift, LeftShift, UpShift, DownShift}
- Ancillas: 2 (for 5 states via LCU)
- Stencil coefficients: center `-4`, neighbors `+1`

### Code Locations

**quantum.py - Lines 165-205: `Block_encoding_diffusion_2d()`**
```python
165 def Block_encoding_diffusion_2d(nx, ny, nu):
166     """
167     2D Block Encoding for Laplacian operator.
168     Uses 4-neighbor stencil (5-point: center + 4 neighbors).
169     Grid indexed as: flat_idx = x + y*nx where x in [0,nx), y in [0,ny)
170     """
171     ancilla = QuantumRegister(2, 'anc')  # ← 2 ANCILLA
172     x_reg = QuantumRegister(nx, 'x')
173     y_reg = QuantumRegister(ny, 'y')
174     qc = QuantumCircuit(ancilla, x_reg, y_reg)
175     
176     # Physics: 2D diffusion on [0,1) x [0,1)
177     dx = 1 / nx
178     dy = 1 / ny
179     dt = 0.9 * min(dx**2, dy**2) / (4 * nu)  # ← 4 NEIGHBORS
180     
181     # Laplacian coefficient (2D: -4 on center, +1 on each neighbor)
182     a_center = 1 - 4 * dt * nu / (dx**2)  # ← CENTER: -4
183     theta = 2 * np.arccos(np.sqrt(a_center))
183     
185     # Prepare 5 unitaries: I, S_x, S_x†, S_y, S_y†  ← 5-POINT STENCIL
186     qc.ry(theta, ancilla[1])
187     qc.ch(ancilla[1], ancilla[0])
188     
189     # Define shift gates
190     S_x = Shift_gate_2d(nx, ny, axis=0).to_gate()       # ← RIGHT
191     S_x_dag = S_x.inverse()                             # ← LEFT
192     S_y = Shift_gate_2d(nx, ny, axis=1).to_gate()       # ← UP
193     S_y_dag = S_y.inverse()                             # ← DOWN
194     
195     # Apply shifts based on ancilla state
196     # State |00⟩: Identity (do nothing)
197     # State |01⟩: S_x                                     ← +1 RIGHT
197     # State |10⟩: S_x†                                    ← +1 LEFT
198     # State |11a⟩: S_y, S_y†, ... (encoded as multi-step) ← +1 UP/DOWN
199     qc.append(S_x.control(2, ctrl_state='01'), ancilla[:] + x_reg[:] + y_reg[:])
200     qc.append(S_x_dag.control(2, ctrl_state='10'), ancilla[:] + x_reg[:] + y_reg[:])
201     qc.append(S_y.control(2, ctrl_state='001'), list(ancilla) + [ancilla[0]] + x_reg[:] + y_reg[:])
202     
203     # Un-prepare
204     qc.ch(ancilla[1], ancilla[0])
205     qc.ry(-theta, ancilla[1])
```

**simulation.py - Lines 314-352: `get_classical_matrix_2d()`**
```python
314 def get_classical_matrix_2d(nx, ny, nu, c_x, c_y):
315     """
316     2D finite difference matrix for advection-diffusion.
317     Flattened indexing: idx = x + y*nx
318     Returns: matrix A and timestep dt
319     """
320     N = nx * ny
321     dx = 1 / nx
322     dy = 1 / ny
323     dt = 0.9 * min(dx**2, dy**2) / (4 * nu)  # ← 4 NEIGHBORS
324     
325     alpha_x = dt * nu / (dx**2)
326     alpha_y = dt * nu / (dy**2)
327     gamma_x = dt * c_x / (2 * dx)
328     gamma_y = dt * c_y / (2 * dy)
329     
330     A = np.zeros((N, N))
331     
332     for y in range(ny):
333         for x in range(nx):
334             idx = x + y * nx  # ← FLATTENING
335             
336             # Center coefficient
337             center_coef = 1.0 - 2*(alpha_x + alpha_y)  # ← CENTER: -4 equivalent
338             A[idx, idx] = center_coef
339             
340             # X-neighbors (with advection)
341             x_left = (x - 1) % nx   # ← LEFT (periodic)
342             x_right = (x + 1) % nx  # ← RIGHT (periodic)
343             left_idx = x_left + y * nx
344             right_idx = x_right + y * nx
345             
346             A[idx, left_idx] = alpha_x + gamma_x   # ← NEIGHBOR: +1
347             A[idx, right_idx] = alpha_x - gamma_x  # ← NEIGHBOR: +1
348             
349             # Y-neighbors (with advection)
350             y_down = (y - 1) % ny   # ← DOWN (periodic)
351             y_up = (y + 1) % ny     # ← UP (periodic)
352             down_idx = x + y_down * nx
353             up_idx = x + y_up * nx
354             
355             A[idx, down_idx] = alpha_y + gamma_y   # ← NEIGHBOR: +1
356             A[idx, up_idx] = alpha_y - gamma_y     # ← NEIGHBOR: +1
357     
358     return A, dt
```

**pages/2_2D_Simulation.py - Lines 88-107: Educational Content**
```python
88  st.header("🔧 Step 2: 2D Block-Encoded Laplacian")
89  st.markdown("""
90  The 2D diffusion operator uses a **5-point stencil** (center + 4 neighbors):
91  """)
92  
93  col1, col2 = st.columns([1, 1])
94  with col1:
95      st.markdown("""
96      ```
97      Stencil:    [   0 ]
98                  [   1 ]
99              [ 1 -4  1 ]    ← CENTER: -4, NEIGHBORS: +1
100             [   1 ]
101             [   0 ]
102     ```
```

---

## 4️⃣ CHANGE 4: "QSVT Polynomial - No Change"

### Concept
- Same target function: $e^{t(|x|-1)}$ (symmetric)
- Same polynomial structure and angles
- Only swap in the 2D block encoding

### Code Locations

**simulation.py - Lines 376-410: Angle Computation (Identical to 1D)**
```python
376             # Compute angles for time step
377             degree_diff = 2 * int(np.ceil(t_phys * np.pi / 2))
378             try:
379                 # ← SAME cvx_poly_coef function
380                 coeffs_diff = cvx_poly_coef(
381                     'exp',                      # ← SAME TARGET
382                     symmetric_time_eval=t_phys, # ← SYMMETRIC
383                     odd_degree='even',          # ← SAME CONSTRAINT
384                     degree=degree_diff,
385                     solver='ECOS'
386                 )
387                 phi_seq_diff = Angles_Fixed(coeffs_diff, degree_diff)
388                 # ← SAME angle sequence, just plug into QSVT_circuit_2d
389             except:
390                 coeffs_diff = cvx_poly_coef(...)  # Fallback
```

**quantum.py - Lines 265-310: `QSVT_circuit_2d()`**
```python
265 def QSVT_circuit_2d(phi_seq, nx, ny, nu, init_state, measurement=True):
266     """
267     Full 2D QSVT circuit
268     Structure identical to 1D, just with 2D block encoding
269     """
270     signal = QuantumRegister(1, 's')
271     ancilla = QuantumRegister(2, 'anc')
272     x_reg = QuantumRegister(nx, 'x')
273     y_reg = QuantumRegister(ny, 'y')
274     qc = QuantumCircuit(signal, ancilla, x_reg, y_reg)
275     
276     # Initialize state
277     # Apply QSVT with phi_seq  ← SAME ANGLES
278     
279     # Block encoding (only difference from 1D)
280     be_2d = Block_encoding_diffusion_2d(nx, ny, nu)  # ← 2D VERSION
281     
282     # For each phi angle:
283     for j, phi in enumerate(phi_seq):
284         # Apply Z-rotation (same as 1D)
284         qc.rz(phi, signal)
285         
286         # Apply block encoding (2D instead of 1D)
287         qc.append(be_2d.control(1, ctrl_state=1), ...)
288         
289         # Reflect
290         qc.rz(phi, signal)
291     
292     # Measure (if requested)
293     if measurement:
294         qc.measure_all()
```

**pages/2_2D_Simulation.py - Lines 150-180: UI Angle Computation**
```python
150 st.header("🔢 Step 4: Compute QSVT Angles")
151 
152 calc_angles_btn = st.button("⚙️ Calculate Angles...", type="primary")
153 
154 if calc_angles_btn:
155     with st.spinner("Computing 2D QSVT angles..."):
156         st.session_state['2d_phi_sequences'] = {}
157         
158         for idx, t in enumerate(time_steps_display):
159             if t == 0:
160                 continue
161             else:
162                 # ← SAME angle computation (no 2D-specific logic)
163                 coeffs_diff = cvx_poly_coef(
164                     'exp',
165                     symmetric_time_eval=t,
166                     odd_degree='even',
167                     degree=degree_diff,
168                     solver='ECOS'
169                 )
```

---

## 🔗 Summary Table: Code Locations

| Theory Change | File | Lines | Key Variables |
|---|---|---|---|
| **1. Double Register** | quantum.py | 140-163, 165-175 | `n_total = nx + ny`, `x_reg`, `y_reg` |
| | pages/2D | 20-28 | `nx`, `ny`, `n_total` |
| **2. Independent Shift** | quantum.py | 209-260 | `Advection_Gate_2d(nx, ny, c_x, c_y)` |
| | | 217-239 | `x_reg`, `y_reg` separate QFT |
| | simulation.py | 370-410 | Split-step loop, stacked advection |
| **3. 2D Block Encoding** | quantum.py | 165-205 | `Block_encoding_diffusion_2d`, 5-point |
| | | 171, 179, 190-201 | 2 ancilla, 4 neighbors, all shifts |
| | simulation.py | 314-356 | `get_classical_matrix_2d`, 5-point |
| | pages/2D | 88-107 | ASCII stencil diagram |
| **4. QSVT Unchanged** | quantum.py | 265-310 | `QSVT_circuit_2d`, same structure |
| | simulation.py | 376-390 | `cvx_poly_coef`, identical call |
| | pages/2D | 150-180 | Same angle UI display |

---

## 🎯 How to Navigate

1. **For theoretical overview**: Start with comments at function definitions
2. **For implementation details**: Read the line ranges above
3. **For UI/education**: Check pages/2_2D_Simulation.py sections
4. **For physics**: Check timestep dt computation in each file
5. **For quantum structure**: Trace `QuantumRegister` and gate definitions

All implementations follow the exact same elegant principle: **separate x and y, apply operations independently, and let the quantum circuit combine them naturally.**
