"""Chay dev server: uvicorn tren 127.0.0.1:8123 (LOCALHOST ONLY).

May nay dang chay cac service production khac (9000/9001/8080) - TUYET DOI
khong bind 0.0.0.0 va khong dung lai cac port do.
"""
import cv2

cv2.setNumThreads(2)

import uvicorn

if __name__ == "__main__":
    uvicorn.run("ai_engine.service.app:app", host="127.0.0.1", port=8123, log_level="info")
