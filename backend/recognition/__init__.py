"""
Recognition package - OCR and face identification.  *Interfaces only.*

Planned modules (Step 3):

    plate_ocr.py    crop -> grayscale -> denoise -> threshold -> EasyOCR ->
                    plate-format post-correction (O/0, I/1 confusions)
    face_recognizer.py
                    RetinaFace detection -> Facenet512 embedding -> cosine
                    similarity against the gallery loaded from
                    `crud.load_face_gallery()` -> Authorized / Unauthorized
"""
