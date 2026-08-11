
---

# 🚀 PHOBOS‑ML — Pull Request

## 📌 Summary
Describe clearly what this PR does.  
Ejemplo: “Implements Binary Cross-Entropy loss with numerical stability.”

---

## 🧩 Related Issue
Closes: #<issue-number>  
Branch: feature/<layer>/<feature-name>

---

## 🔍 Changes Introduced
- [ ] New header declarations  
- [ ] New `.c` file(s)  
- [ ] Modifications to existing modules  
- [ ] New tests  
- [ ] CI updates  
- [ ] Documentation updates  

Explain each change briefly:

**Headers**
- Added `tensor_bce_loss` declaration.

**Source Files**
- Created `ops_loss.c` with BCE implementation.

**Tests**
- Added `tests/test_bce.c`.

**CI**
- CI auto-detects and runs new tests.

---

## 🧪 Test Coverage
List the tests added or affected:

- `test_bce` — validates BCE numerical stability  
- All tests run automatically via CI

If applicable, describe manual tests or edge cases validated.

---

## 🛡️ Stability & Safety
Explain how numerical stability, memory safety, or edge cases were handled.

Examples:
- Clamped predictions to avoid `log(0)`  
- Verified shape compatibility  
- Ensured no memory leaks (Valgrind clean)

---

## 📂 Affected Modules
Mark the modules touched by this PR:

- [ ] `tensor.c`
- [ ] `ops.c`
- [ ] `ops_activation.c`
- [ ] `ops_loss.c`
- [ ] `ops_grad.c`
- [ ] `Makefile`
- [ ] `CI`
- [ ] `tests/`

---

## 🧭 Layer Target
Indicate the layer this PR is intended for:

- [ ] `dev`
- [ ] `qa`
- [ ] `main`

---

## 📸 Screenshots / Logs (Optional)
Paste relevant logs, Valgrind output, or CI results.

---

## ✔ Reviewer Checklist
(For maintainers)

- [ ] Code compiles  
- [ ] Tests pass  
- [ ] CI passes  
- [ ] No memory leaks  
- [ ] No formatting issues  
- [ ] Documentation updated  
- [ ] Branch naming follows NA‑Engine style  
- [ ] PR is scoped and clean  

---

## 📝 Additional Notes
Anything else reviewers should know.
```

---

