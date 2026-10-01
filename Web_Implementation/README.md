# Brain Tumor Screening — Prototipe Lokal

## 1. Struktur folder
```
brain-tumor-app/
├── app.py
├── requirements.txt
├── repvit_brain_tumor_best.pt   <- taruh file model kamu di sini
└── templates/
    └── index.html
```

## 2. Buat virtual environment (opsional tapi disarankan)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

## 3. Install dependencies
```bash
pip install -r requirements.txt
```
Catatan: `requirements.txt` menunjuk ke index PyTorch CPU. Kalau kamu punya GPU CUDA dan ingin pakai GPU, install torch versi CUDA secara terpisah dulu (lihat pytorch.org) sebelum `pip install -r requirements.txt`.

## 4. Taruh file model
Copy `repvit_brain_tumor_best.pt` ke folder `brain-tumor-app/` (sejajar dengan `app.py`). Nama file harus sama persis, atau ubah `MODEL_PATH` di `app.py`.

## 5. Jalankan
```bash
python app.py
```
Buka browser ke: **http://localhost:5000**

## 6. Menghentikan server
Tekan `Ctrl+C` di terminal.

## Troubleshooting
- **`FileNotFoundError: repvit_brain_tumor_best.pt`** → file model belum ada di folder yang benar.
- **`ModuleNotFoundError: No module named 'timm'`** dll → jalankan ulang `pip install -r requirements.txt` di virtual environment yang aktif.
- **Port 5000 sudah dipakai** → ubah baris terakhir `app.py` jadi `app.run(host="0.0.0.0", port=5001, debug=True)` lalu akses `http://localhost:5001`.
- **Loading pertama lambat** → normal, model PyTorch di-load sekali saat server start.
