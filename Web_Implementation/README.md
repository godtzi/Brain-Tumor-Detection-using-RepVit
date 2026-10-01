In Indonesian Language
Will add English Language, for now use AI to translate

# Brain Tumor Screening  Prototipe Lokal

## 1. Struktur folder
```
brain-tumor-app/
├── app.py
├── requirements.txt
├── repvit_brain_tumor_best.pt
└── templates/
    └── index.html
```

## 2. Install dependencies
```bash
pip install -r requirements.txt
```
Catatan: `requirements.txt` menunjuk ke index PyTorch CPU. Kalau kamu punya GPU CUDA dan ingin pakai GPU, install torch versi CUDA secara terpisah dulu (lihat pytorch.org) sebelum `pip install -r requirements.txt`.

## 3. Jalankan
```bash
python app.py
```
Buka browser ke: **http://localhost:5000**

## 4. Menghentikan server
Tekan `Ctrl+C` di terminal.

## Troubleshooting
- **`FileNotFoundError: repvit_brain_tumor_best.pt`** → file model belum ada di folder yang benar atau Lokasi perlu diubah sesuai dengan lokasi dari model.
- **`ModuleNotFoundError: No module named 'timm'`** dll → jalankan ulang `pip install -r requirements.txt`
- **Port 5000 sudah dipakai** → ubah baris terakhir `app.py` jadi `app.run(host="0.0.0.0", port=5001, debug=True)` lalu akses `http://localhost:5001`.
- **Loading pertama lambat** → normal, model PyTorch di-load sekali saat server start.
