# ai_service/app/services/pdf_service.py

import io
import PyPDF2
import fitz  # PyMuPDF
from PIL import Image
from fastapi import HTTPException
from typing import List, Optional

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """
    Bir PDF dosyasının ham baytlarını (in-memory) alır ve metnini çıkarır.
    Misafir kullanıcıların senkron istekleri için kullanılır.
    """
    try:
        # Bayt verisini bellekte bir dosya gibi aç
        pdf_file = io.BytesIO(pdf_bytes)
        
        # PyPDF2 ile oku
        reader = PyPDF2.PdfReader(pdf_file)
        
        text_parts = []
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text_parts.append(extracted)
        
        full_text = "\n".join(text_parts)
        
        if not full_text.strip():
            # Bu durum genellikle taranmış (scanned) PDF'lerde olur
            raise HTTPException(
                status_code=400, 
                detail="PDF'ten metin çıkarılamadı. Dosya taranmış bir resim olabilir."
            )
            
        return full_text

    except PyPDF2.errors.PdfReadError:
        raise HTTPException(status_code=400, detail="Geçersiz veya bozuk PDF dosyası.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF işleme hatası: {str(e)}")


def extract_text_from_pdf_path(storage_path: str) -> str:
    """
    Paylaşılan volume'deki bir PDF dosyasının yolunu alır ve metnini çıkarır.
    Kayıtlı kullanıcıların asenkron Celery görevleri için kullanılır.
    """
    try:
        # Dosyayı paylaşılan diskten (shared_uploads) aç
        with open(storage_path, "rb") as pdf_file:
            
            # PyPDF2 ile oku
            reader = PyPDF2.PdfReader(pdf_file)
            
            text_parts = []
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text_parts.append(extracted)
            
            full_text = "\n".join(text_parts)

            if not full_text.strip():
                raise HTTPException(
                    status_code=400, 
                    detail="PDF'ten metin çıkarılamadı (taranmış resim)."
                )
                
            return full_text

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dosya bulunamadı: {storage_path}")
    except PyPDF2.errors.PdfReadError:
        raise HTTPException(status_code=400, detail="Geçersiz veya bozuk PDF dosyası.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF işleme hatası: {str(e)}")


def extract_images_from_pdf_bytes(
    pdf_bytes: bytes,
    max_pages: Optional[int] = None,
    dpi: int = 150
) -> List[bytes]:
    """
    PDF'den görüntüleri çıkarır (sayfaları görüntüye çevirir).
    Taranmış PDF'ler veya görsel içerikli PDF'ler için kullanılır.
    
    Args:
        pdf_bytes: PDF dosyasının ham baytları
        max_pages: Maksimum sayfa sayısı (None = tüm sayfalar)
        dpi: Görüntü çözünürlüğü (varsayılan: 150)
    
    Returns:
        Görüntü baytları listesi (PNG formatında)
    """
    try:
        pdf_file = io.BytesIO(pdf_bytes)
        doc = fitz.open(stream=pdf_file, filetype="pdf")
        
        images = []
        page_count = min(len(doc), max_pages) if max_pages else len(doc)
        
        for page_num in range(page_count):
            page = doc[page_num]
            
            # Sayfayı görüntüye çevir (matris ile DPI ayarı)
            mat = fitz.Matrix(dpi / 72, dpi / 72)  # 72 DPI varsayılan
            pix = page.get_pixmap(matrix=mat)
            
            # PIL Image'e çevir
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # PNG formatında bytes'a çevir
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="PNG")
            images.append(img_bytes.getvalue())
        
        doc.close()
        
        if not images:
            raise HTTPException(
                status_code=400,
                detail="PDF'ten görüntü çıkarılamadı."
            )
        
        return images
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"PDF görüntü çıkarma hatası: {str(e)}"
        )


def extract_images_from_pdf_path(
    storage_path: str,
    max_pages: Optional[int] = None,
    dpi: int = 150
) -> List[bytes]:
    """
    Paylaşılan volume'deki PDF'den görüntüleri çıkarır.
    
    Args:
        storage_path: PDF dosyasının yolu
        max_pages: Maksimum sayfa sayısı
        dpi: Görüntü çözünürlüğü
    
    Returns:
        Görüntü baytları listesi
    """
    try:
        with open(storage_path, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()
            return extract_images_from_pdf_bytes(pdf_bytes, max_pages, dpi)
            
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dosya bulunamadı: {storage_path}")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"PDF görüntü çıkarma hatası: {str(e)}"
        )