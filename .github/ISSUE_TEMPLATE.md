
---

# 🧩 PHOBOS‑ML — Issue Template

## 📌 Issue Type
Selecciona el tipo de issue:
- [ ] Enhancement
- [ ] Bug
- [ ] Feature
- [ ] Refactor
- [ ] Documentation
- [ ] CI / DevOps
- [ ] Research / Design

---

## 📝 Summary
Describe brevemente el objetivo del issue.  
Ejemplo: “Implement Binary Cross-Entropy loss with numerical stability.”

---

## 🎯 Acceptance Criteria
Lista clara de condiciones que deben cumplirse para cerrar el issue.

Ejemplo:
- Stable for extreme values
- Tests included
- No memory leaks
- CI passes in `dev` and `qa`

---

## 🧠 Technical Description
Explica qué se debe implementar o corregir.

Incluye:
- API propuesta (headers)
- Archivos `.c` involucrados
- Algoritmos o fórmulas
- Consideraciones de estabilidad numérica
- Manejo de errores

---

## 📂 Affected Modules
Marca los módulos que serán modificados:

- [ ] `tensor.h`
- [ ] `tensor.c`
- [ ] `ops.c`
- [ ] `ops_activation.c`
- [ ] `ops_loss.c`
- [ ] `ops_grad.c`
- [ ] `Makefile`
- [ ] `CI`
- [ ] `tests/`
- [ ] `docs/`

---

## 🧪 Testing Requirements
Define los tests necesarios:

- Nombre del archivo de test
- Casos básicos
- Casos extremos
- Casos de error
- Validación numérica (tolerancia)

Ejemplo:
- `tests/test_bce.c`
- Test for clamping near 0 and 1
- Test for shape mismatch

---

## 🛡️ Stability & Safety Requirements
Indica requisitos de seguridad:

- [ ] Numerical stability
- [ ] Memory safety (Valgrind clean)
- [ ] Shape validation
- [ ] No undefined behavior
- [ ] No leaks on error paths

---

## 🧭 Target Layer
Indica la capa donde se trabajará:

- [ ] `dev`
- [ ] `qa`
- [ ] `main`

---

## 🔗 Related PRs / Issues
Lista issues o PRs relacionados.

Ejemplo:
- Depends on: #12
- Will be used by: #14

---

## 📎 Additional Notes
Cualquier información extra que ayude al desarrollo o revisión.

---
