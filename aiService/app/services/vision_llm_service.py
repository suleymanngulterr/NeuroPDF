import os
import base64
import io
import ollama
from typing import List, Dict, Optional

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llava")

def image_to_base64(image_bytes: bytes) -> str:
    """Görüntüyü base64 formatına çevirir"""
    return base64.b64encode(image_bytes).decode('utf-8')

def analyze_image_with_vision_llm(
    image_bytes: bytes,
    prompt: str = "Bu görüntüyü detaylı bir şekilde açıkla. Türkçe cevap ver.",
    model: Optional[str] = None
) -> Dict[str, str]:
    """
    Görüntüyü vision LLM ile analiz eder.
    
    Args:
        image_bytes: Görüntü baytları (PNG, JPEG, vb.)
        prompt: Analiz için prompt
        model: Kullanılacak model (varsayılan: llava)
    
    Returns:
        {"description": "görüntü açıklaması"}
    """
    client = ollama.Client(host=OLLAMA_HOST)
    model = model or OLLAMA_VISION_MODEL
    
    try:
        # Ollama vision API'si için görüntüyü direkt bytes olarak gönderebiliriz
        # veya base64 formatında gönderebiliriz
        # Ollama Python client'ı her iki formatı da destekler
        
        response = client.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_bytes]  # Ollama direkt bytes kabul eder
                }
            ],
            options={"temperature": 0.3},
        )
        
        description = response["message"]["content"]
        return {"description": description}
        
    except Exception as e:
        # Eğer bytes çalışmazsa base64 dene
        try:
            image_base64 = image_to_base64(image_bytes)
            response = client.chat(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [image_base64]
                    }
                ],
                options={"temperature": 0.3},
            )
            description = response["message"]["content"]
            return {"description": description}
        except Exception as e2:
            return {"description": f"Vision LLM hatası: {str(e2)}"}

def analyze_multiple_images_with_vision_llm(
    images: List[bytes],
    prompt: str = "Bu görüntüleri analiz et ve aralarındaki ilişkiyi açıkla. Türkçe cevap ver.",
    model: Optional[str] = None
) -> Dict[str, str]:
    """
    Birden fazla görüntüyü birlikte analiz eder.
    
    Args:
        images: Görüntü baytları listesi
        prompt: Analiz için prompt
        model: Kullanılacak model
    
    Returns:
        {"description": "görüntülerin analizi"}
    """
    client = ollama.Client(host=OLLAMA_HOST)
    model = model or OLLAMA_VISION_MODEL
    
    try:
        # Ollama direkt bytes listesi kabul eder
        response = client.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": images  # Direkt bytes listesi
                }
            ],
            options={"temperature": 0.3},
        )
        
        description = response["message"]["content"]
        return {"description": description}
        
    except Exception as e:
        # Eğer bytes çalışmazsa base64 dene
        try:
            image_base64_list = [image_to_base64(img) for img in images]
            response = client.chat(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "images": image_base64_list
                    }
                ],
                options={"temperature": 0.3},
            )
            description = response["message"]["content"]
            return {"description": description}
        except Exception as e2:
            return {"description": f"Vision LLM hatası: {str(e2)}"}
