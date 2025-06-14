# Open-Source Large Language Model Development

## Overview

This project focuses on reimplementing the development cycle of Large Language Models (LLMs) to advance open-source base models for specialized applications. The goal is to transition from being an LLM user to an LLM builder by enhancing model capabilities through key techniques: Supervised Fine-Tuning (SFT), GPRO-based Reinforcement Learning, multimodal integration (vision and audio encoders), quantization for edge deployment, and continued pre-training for domain-specific knowledge.

The project is conducted under the guidance of **Dr. NR Sunitha**, Professor & Head, Department of Computer Science and Engineering, Siddaganga Institute of Technology (SIT), Tumakuru, by the **Batch of 2026**:

- **Pavan J** (1SI22CI035)
- **Chaithra HR** (1SI22CI010)
- **Ayush Shankar Prasad** (1SI22CI009)

**Department**: CSE (AIML), SIT Tumakuru\
**Batch**: B25

---

## Problem Statement

Large Language Model (LLM) engineering is dominated by Big Tech companies with vast computational and financial resources, limiting opportunities for small teams with constrained resources to contribute to foundational model development. This project aims to explore LLM engineering using limited resources to create efficient, high-performing, and versatile open-source models.

---

## Objectives

1. **Supervised Fine-Tuning (SFT)**: Fine-tune a base model to create an instruct model capable of answering questions and following instructions.
2. **Reasoning Model**: Apply GPRO-based Reinforcement Learning to enhance logical reasoning and thinking capabilities.
3. **Multimodal Model**: Integrate vision (SigLIP, CLIP) and audio (Whisper, Saaras) encoders to enable multimodal functionality.
4. **Quantization**: Reduce model weights to lower precision (4-bit/8-bit) for efficient edge deployment.
5. **Continued Pre-training**: Incorporate domain-specific data to enhance model adaptability for specialized tasks.

---

## Methodology

1. **Base Model Selection**: Choose an open-source LLM (e.g., Gemma 3, Phi 4, Llama 3) based on architecture and performance suitability.
2. **Supervised Fine-Tuning (SFT)**: Train the base model with labeled prompt-answer datasets to produce an instruct model.
3. **Reasoning Enhancement**: Use GPRO-based Reinforcement Learning to improve logical inference by generating responses, scoring them with a reward function, and training the model based on these scores.
4. **Multimodal Integration**: Attach vision and audio encoders using projection layers to enable text, image, and audio processing.
5. **Quantization**: Convert model weights to lower precision (int8, int4) to reduce memory and compute requirements.
6. **Continued Pre-training**: Train the base model with additional unlabeled domain-specific data to enhance knowledge.

---

## System Architecture

- **Continued Pre-training**: Train an existing base model with additional raw data to incorporate new knowledge.
- **Supervised Fine-Tuning**: Use labeled datasets to create an instruct model capable of answering questions.
- **Reasoning Model (GPRO)**: Generate responses, score them with a reward function, and use scores to train the model for reasoning.
- **Multimodal Model**: Integrate mode-specific encoders (vision/audio) with projection layers for multimodal functionality.
- **Quantization**: Convert high-precision weights (F32, F16) to lower-precision formats (int8, int4) for efficient deployment.

---

## Tools and Technologies

### Compute

- **SIT-GPU (2x L40)**: For code and process validation.
- **Lightning AI (H100)**: For model training.

### Base Models

- Gemma 3, Phi 4, Llama 3 families.

### Teacher Models

- DeepSeek, Llama 4, Gemini 2 families.

### Datasets

- Hugging Face Datasets, Bhashini Project, AI4Bharat.

### Low-Level ML & Processing Tools

- CUDA, PyTorch, Flash Attention 2.

### High-Level Tools

- **Training**: Unsloth AI, Hugging Face Transformers, NVIDIA NeMo (for continued pre-training, SFT, GPRO).
- **Quantization**: Hugging Face Transformers, NVIDIA NeMo.
- **Inference**: Transformers, vLLM, TensorRT-LLM.

### Multimodal Encoders

- **Vision**: SigLIP, CLIP.
- **Audio**: Whisper, Saaras.

---

## Design Solution

- **Scalability**: Modular architecture (LoRA) for seamless updates and integration of components (vision, audio, text).
- **Efficiency**: Quantization (4-bit/8-bit) to reduce computational requirements for edge deployment.
- **Performance Optimization**: Knowledge distillation to transfer capabilities from larger teacher models to smaller student models for faster inference.
- **Tools & Frameworks**: HuggingFace for model hosting, NVIDIA NeMo for training/deployment, Unsloth AI for optimized inference.
- **Evaluation & Monitoring**: Continuous evaluation using benchmark datasets to ensure model reliability.

---

## Literature Survey

- **Supervised Fine-Tuning (SFT)**: Effective for adapting pretrained LLMs for tasks like instruction following and code generation.
- **Knowledge Distillation**: Transfers capabilities from larger teacher models to smaller student models.
- **Multimodal Integration**: Models like LLaVA and Phi 4 demonstrate modularity by attaching vision/audio encoders to text LLMs.
- **Quantization**: 4-bit and 8-bit quantization reduces memory and compute needs for edge deployment.
- **Tool Calling**: Enables LLMs to interact with external tools and APIs for agentic capabilities.
- **Reasoning Abilities**: Techniques like chain-of-thought prompting and program-aided reasoning improve logical capabilities.
- **Open-Source LLMs**: Diverse architectures influence fine-tuning outcomes.
- **Edge Deployment**: Quantization and efficient inference libraries make edge deployment feasible.

---

## 