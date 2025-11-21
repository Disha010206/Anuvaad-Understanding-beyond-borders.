# Anuvaad – Multilingual Customer Support Assistant  
### *"Understanding beyond borders."*

Anuvaad is an AI-powered multilingual support system that can **detect languages**, **translate customer queries**, **classify intents**, and generate **professional customer support replies**—all inside a clean and elegant UI.

This project is built using **Flask + HuggingFace Transformers + NLLB-200** and works for *200+ languages* including Indian, Asian, Middle-Eastern & European languages.

---

## Features

### **AI-Powered Processing**
- Detects the language of the input text automatically  
- Translates any language to **English** using **NLLB-200 (600M)**  
- Classifies customer intent  
- Generates a helpful **suggested support reply**  
- Tracks latency & usage metrics for every query

### **Modern UI**
- Clean sage-green theme  
- Fully responsive  
- 4 neatly arranged cards: Input, Snapshot, Understanding, Reply  
- Smooth shadows, rounded UI, and elegant typography  
- Custom logo based on Hindi letter **“अ / अ़” (A)** + beige theme

### **Session Metrics**
- Total queries processed  
- Average latency  
- Languages encountered  
- Feedback counter

### **Local, Private & Free**
Everything runs **locally on your machine** using open-source HF models.  
No API keys. No charges. No data leaves your system.

---

## Tech Stack

| Layer | Technology |
|------|------------|
| Backend | Flask (Python) |
| Translation | NLLB-200-Distilled-600M |
| NLP | HuggingFace Transformers |
| Frontend | HTML, CSS, JS |
| Intent Detection | BART-MNLI |
| Reply Generation | FLAN-T5-Base |


