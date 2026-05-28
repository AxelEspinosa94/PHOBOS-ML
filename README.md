
---

# PHOBOS-ML Project

# PHOBOS Machine Learning Engine  
[![Language](https://img.shields.io/badge/language-C-blue.svg)]()
[![Build](https://img.shields.io/badge/build-Makefile-orange.svg)]()
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20WSL2-green.svg)]()
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)]()
[![Status](https://img.shields.io/badge/status-Active-success.svg)]()

PHOBOS-ML is a lightweight **Machine Learning engine written in pure C**, designed with a focus on:

- **Modularity** — clean separation between tensors, operations, models, and optimizers  
- **Performance** — minimal overhead, close to the metal  
- **Interoperability** — built to integrate with Python (ctypes/cffi) and Java (JNI)  
- **Extensibility** — structured to grow into a full ML inference engine  

This project is part of a broader ecosystem:

- **A)** PHOBOS — Core ML engine in C  
- **B)** Spark‑ML Java Project — Custom Transformers & Estimators for MLlib  
- **C)** Hybrid Engine — C + Python + Java + Spark integration layer  

---

## 📁 Project Structure

```
PHOBOS-ML/
│
├── include/               # Public headers (API)
│   ├── tensor.h
│   ├── ops.h
│   ├── model.h
│   └── optimizer.h
│
├── src/                   # Implementation
│   ├── tensor.c
│   ├── ops.c
│   ├── model.c
│   └── optimizer.c
│
├── tests/                 # Unit tests
│   ├── test_tensor.c
│   ├── test_ops.c
│   └── test_model.c
│
├── examples/              # Usage examples
│   ├── example_lr.c
│   └── example_tensor_ops.c
│
├── build/                 # Compiled artifacts (ignored by Git)
│
├── Makefile               # Main build script
├── README.md              # Documentation
└── .gitignore
```

---

## 🚀 Build Instructions

Compile the static library:

```bash
make
```

Clean build artifacts:

```bash
make clean
```

---

## 🧪 Running Tests

Example (manual compilation):

```bash
gcc tests/test_tensor.c src/tensor.c -Iinclude -o test_tensor
./test_tensor
```

You may add a full test runner later.

---

## 📜 License (MIT)

This project is licensed under the **MIT License**, a permissive open‑source license that allows commercial and private use.

```
MIT License

Copyright (c) 2026 Axel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 🧩 Roadmap

- [ ] Implement `tensor.c` (core tensor structure)
- [ ] Add basic operations (MatMul, ReLU, Softmax)
- [ ] Implement Logistic Regression model
- [ ] Add SGD optimizer
- [ ] Build shared library `.so` for Python bindings
- [ ] Create JNI wrapper for Java
- [ ] Integrate with Spark MLlib (Transformer + Estimator)
- [ ] Add benchmarks and profiling tools

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome.  
Feel free to open a PR or start a discussion.

---

## ⭐ Acknowledgements

This project is inspired by the internals of frameworks such as:

- PyTorch (C++ backend)
- TensorFlow (C++ core)
- ONNX Runtime
- XGBoost / LightGBM

PHOBOS-ML aims to provide an educational yet production‑oriented view of how ML engines work under the hood.


---

# 🚀 VSCode + Ubuntu (WSL) Integration — Quick Guide

When you have Ubuntu working, here’s the clean workflow:

### **1. Install the VSCode extension**
In VSCode:

- Go to **Extensions**
- Install **WSL** extension (Approved by Microsoft)

### **2. Open Ubuntu in VSCode**
In Windows:

- Open VSCode  
- Press `Ctrl + Shift + P`  
- Run:

```
WSL: Connect to WSL using distro
```

and select Ubuntu

### **3. Open your project folder**
In Ubuntu terminal:

```bash
cd ~/PHOBOS-ML
code .
```

VSCode will open **inside Ubuntu**, using Linux compilers, Linux paths, and Linux terminals.

### **4. Install Git (if not installed)**
```bash
sudo apt install git
```

Configure:

```bash
git config --global user.name <user-name>
git config --global user.email <user-email>
```

### **5. Push to GitHub**
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/youruser/PHOBOS-ML.git
git push -u origin main
```

---
