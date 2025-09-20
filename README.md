# Intelligent Chatbot Architectures

This repository demonstrates the evolution of chatbot development using three major approaches:
1. **Deep Learning models (BiLSTM + GRU)**  
2. **Transformer-based models (ALBERT, DistilBERT)**  
3. **Large Language Models with Retrieval-Augmented Generation (LangChain + GPT)**  

It showcases progression from classical intent classification to state-of-the-art conversational AI.

---

## 📂 Repository Structure
```

intelligent-chatbot-architectures/
│── deep\_learning\_chatbot/    # BiLSTM & GRU intent classification
│── transformer\_chatbot/      # Fine-tuned BERT, ALBERT, DistilBERT
│── gpt\_chatbot/              # GPT-2 based chatbot
│── rag\_chatbot/              # LangChain RAG chatbot with FAISS

````

---

## 🚀 Techniques Covered

### 🔹 Deep Learning Chatbots  
- Bidirectional LSTM + GRU hybrid  
- Pre-trained GloVe embeddings  
- Task: Intent classification  

### 🔹 Transformer Chatbots  
- **BERT**: Healthcare/Medicare chatbot  
- **ALBERT & DistilBERT**: Lightweight transformer models  
- Hugging Face Transformers  

### 🔹 GPT-based Chatbot  
- GPT-2 fine-tuned conversational model  

### 🔹 RAG-based Chatbot (LangChain)  
- Knowledge-grounded chatbot with **FAISS vector DB**  
- Embeddings: OpenAI `text-embedding-ada-002`  
- Backend: LangChain + GPT-3.5  

---



## 🏋️ Training & Running

### Deep Learning

```bash
cd deep_learning_chatbot
python train_gru_lstm.py
```

### Transformers

```bash
cd transformer_chatbot
python train_distilbert.py
```

### GPT-2 Chatbot

```bash
cd gpt_chatbot
python gpt2_train.py
```

### RAG Chatbot

```bash
cd rag_chatbot
python rag_chatbot.py
```

---

## 📦 Requirements

* TensorFlow / PyTorch
* Hugging Face Transformers
* LangChain, FAISS
* scikit-learn, numpy, pandas

---

## 📌 Notes

* Each sub-project highlights trade-offs in performance, scalability, and data requirements.
* The RAG-based chatbot demonstrates state-of-the-art knowledge-grounded conversations.

---
