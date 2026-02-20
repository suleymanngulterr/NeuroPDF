# NeuroPDF Local LLM Kurulum ve Kullanım Kılavuzu

## ✅ Kurulum Tamamlandı!

Local LLM (Ollama) başarıyla kuruldu ve test edildi.

## 📋 Yapılanlar

1. ✅ Ollama servisi kuruldu ve çalışıyor
2. ✅ `phi3:mini` modeli indirildi (2.2 GB)
3. ✅ Docker Compose yapılandırması güncellendi
4. ✅ Test scriptleri oluşturuldu

## 🚀 Kullanım

### 1. Local LLM'i Test Etme

Basit test scripti ile test edebilirsiniz:

```bash
cd ~/Projects/NeuroPDF
python3 test_local_llm_simple.py
```

### 2. Docker ile Çalıştırma

Docker Compose ile tüm servisleri başlatmak için:

```bash
cd ~/Projects/NeuroPDF
docker-compose up -d
```

**Not:** İlk çalıştırmada Ollama container'ı içinde modeli indirmeniz gerekebilir:

```bash
docker exec -it pdf_designer_llm ollama pull phi3:mini
```

### 3. Backend'de Local LLM Kullanma

Backend API'de local LLM kullanmak için `llm_provider='local'` parametresini kullanın:

```python
# Örnek: Metin özetleme
result = summarize_text(
    text="Metin içeriği...",
    prompt_instruction="Özetle",
    llm_provider="local"  # 'cloud' yerine 'local'
)

# Örnek: PDF Chat
result = chat_over_pdf(
    session_text="PDF içeriği...",
    filename="dosya.pdf",
    history_text="Sohbet geçmişi...",
    user_message="Kullanıcı sorusu",
    llm_provider="local"  # 'cloud' yerine 'local'
)
```

## ⚙️ Yapılandırma

### Ortam Değişkenleri

AI Service için `.env` dosyasında (opsiyonel):

```env
OLLAMA_HOST=http://ollama:11434  # Docker içinde
# veya
OLLAMA_HOST=http://localhost:11434  # Yerel geliştirme için

OLLAMA_MODEL=phi3:mini
```

### Farklı Model Kullanma

Daha güçlü bir model kullanmak isterseniz:

```bash
# Model indir
ollama pull llama3.2:3b  # veya başka bir model

# Docker Compose'da model adını değiştir
# docker-compose.yml içinde OLLAMA_MODEL değerini güncelle
```

Popüler modeller:
- `phi3:mini` (2.2 GB) - Hızlı, küçük
- `llama3.2:3b` (2.0 GB) - İyi performans
- `mistral:7b` (4.1 GB) - Daha güçlü
- `qwen2.5:7b` (4.4 GB) - Çok dilli destek

## 🔍 Sorun Giderme

### Ollama servisi çalışmıyor

```bash
# Servisi başlat
ollama serve

# Veya arka planda
nohup ollama serve > /dev/null 2>&1 &
```

### Model bulunamıyor

```bash
# Yüklü modelleri listele
ollama list

# Model indir
ollama pull phi3:mini
```

### Docker container'da model yok

```bash
# Container'a gir
docker exec -it pdf_designer_llm bash

# Model indir
ollama pull phi3:mini
```

## 📝 Notlar

- **phi3:mini** modeli Türkçe desteği sınırlıdır. Daha iyi Türkçe için `qwen2.5:7b` veya `mistral:7b` önerilir.
- Local LLM kullanırken GPU varsa performans önemli ölçüde artar.
- Model boyutu ve performans arasında denge kurun (daha büyük model = daha iyi sonuç ama daha yavaş).

## 🎯 Sonraki Adımlar

1. Daha büyük bir model deneyin (performans için)
2. GPU desteğini aktifleştirin (varsa)
3. Production için model seçimini optimize edin
4. Türkçe için daha uygun bir model seçin

---

**Test Tarihi:** 20 Ocak 2025  
**Model:** phi3:mini  
**Durum:** ✅ Çalışıyor
