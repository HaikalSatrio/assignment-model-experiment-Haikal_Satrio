# AI Model Experiment & Evaluation

**Individual Assignment: Sentiment Analysis Ulasan Pelanggan E-commerce**

**Nama Peserta:** Haikal Satrio
**Tanggal eksperimen:** 14 November 2026
**Repository:** https://github.com/HaikalSatrio/assignment-model-experiment-Haikal_Satrio

---

## 1. Ringkasan

Project ini membandingkan dua pendekatan untuk mengklasifikasikan sentimen ulasan pelanggan e-commerce ke dalam dua kelas, yaitu `positif` dan `negatif`:

1. **Model klasik Machine Learning:** TF-IDF + Logistic Regression dari Scikit-learn.
2. **LLM API:** Gemini 3.1 Flash Lite dengan prompt zero-shot.

Keduanya diuji pada test set yang sama agar perbandingannya adil. Selain test set 8 kalimat, dilakukan perbandingan tambahan pada seluruh 40 kalimat unik memakai 5-fold cross-validation untuk model klasik dan inference penuh untuk LLM.

**Hasil singkat:** pada 40 kalimat unik, model klasik memperoleh accuracy dan F1 macro 0,70, sedangkan Gemini 3.1 Flash Lite memperoleh 1,00. Pada test set 8 kalimat, keduanya memperoleh 1,00 sehingga hasilnya seri.

---

## 2. Problem Statement

### 2.1 Objective

Membangun fitur otomatis yang mengklasifikasikan sentimen ulasan pelanggan e-commerce ke dalam dua kelas, positif atau negatif. Dengan fitur ini, ulasan negatif dapat difilter atau dieskalasi ke tim Customer Service tanpa harus dibaca manual satu per satu.

Eksperimen ini membandingkan dua kandidat pendekatan sebelum tim memilih satu untuk dikembangkan di sistem produksi.

### 2.2 Target atau label yang diprediksi

| Item | Keterangan |
|---|---|
| Input | `review_text`, teks ulasan pelanggan berbahasa Indonesia |
| Target | `sentiment`, label biner `positif` atau `negatif` |
| Task | Binary text classification atau sentiment analysis |
| Sumber label | Sudah tersedia di dataset, termasuk supervised learning |
| Metrik utama | F1 macro, dengan accuracy sebagai pembanding |

### 2.3 Batasan dan asumsi

1. Dataset berisi 200 baris, tetapi hanya 40 kalimat unik. Setiap kalimat diulang sekitar 5 kali.
2. Domain e-commerce dengan bahasa Indonesia informal, banyak singkatan dan typo.
3. Hanya ada dua kelas. Tidak ada kelas netral, sehingga ulasan ambigu dipaksa masuk salah satu kelas.
4. Sarkasme dan negasi kompleks tidak dimodelkan secara khusus.
5. Evaluasi dilakukan pada kalimat yang tidak dilihat model saat training, bukan pada data produksi.
6. Ukuran test set kecil, yaitu 8 kalimat, sehingga hasil test set dilengkapi dengan 5-fold cross-validation pada 40 kalimat.
7. Baseline pembanding adalah 0,50 karena setelah deduplikasi komposisi kelas seimbang 20:20.

---

## 3. Dataset

File: `data/customer_reviews_sentiment.csv`

| Kolom | Keterangan |
|---|---|
| `review_id` | ID ulasan |
| `product_name` | Nama produk |
| `review_text` | Teks ulasan pelanggan |
| `sentiment` | Label sentimen, `positif` atau `negatif` |

Karakteristik dataset:

- Jumlah baris: 200
- Jumlah kalimat unik: 40
- Duplikasi: setiap kalimat muncul sekitar 5 kali
- Distribusi label data mentah: 110 positif dan 90 negatif
- Distribusi label setelah deduplikasi: 20 positif dan 20 negatif
- Panjang teks: 43 sampai 71 karakter
- Dataset asli tidak diubah. Deduplikasi hanya dilakukan di memori saat notebook dijalankan.

Contoh baris:

| review_id | product_name | review_text | sentiment |
|---|---|---|---|
| 1 | Kemeja Flanel | Pelayanan lambat dan tidak responsif saat dikomplain. | negatif |
| 2 | Case HP | Puas banget belanja disini, proses cepat dan barang aman. | positif |

### 3.1 Mengapa deduplikasi penting

Tanpa deduplikasi, kalimat yang sama dapat muncul di train dan test sekaligus. Model akan terlihat sangat akurat karena hanya menghafal kalimat yang sudah pernah dilihat. Ini disebut **data leakage** dan membuat metrik tidak valid.

Karena itu notebook melakukan:

```python
df = df.drop_duplicates(subset='review_text').reset_index(drop=True)
```

Hasilnya 40 kalimat unik yang kemudian dibagi menjadi:

- Train: 32 kalimat
- Test: 8 kalimat
- Proporsi kelas dijaga seimbang dengan `stratify=y`
- `random_state=42` agar hasil dapat direproduksi

---

## 4. Struktur Project

```text
assignment-model-experiment-Haikal_Satrio/
├── data/
│   └── customer_reviews_sentiment.csv
├── documentation/
│   ├── comparison_table.csv
│   ├── comparison_table_40.csv
│   ├── confusion_matrix_classic_test.csv
│   ├── confusion_matrix_llm_test.csv
│   ├── confusion_matrix_classic_cv_40.csv
│   ├── confusion_matrix_llm_40.csv
│   └── model_comparison_summary.png
├── notebook/
│   └── experiment_notebook.ipynb
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 5. Metodologi

### 5.1 Pendekatan 1: model klasik Scikit-learn

Pipeline:

1. Teks diubah menjadi fitur numerik memakai **TF-IDF**.
2. Model **Logistic Regression** dilatih pada 32 kalimat training.
3. Prediksi dilakukan pada 8 kalimat test.
4. Validasi tambahan memakai **5-fold Stratified Cross-Validation** pada seluruh 40 kalimat unik.

TF-IDF bekerja dengan memberi bobot pada setiap kata:

- **TF (Term Frequency):** seberapa sering kata muncul dalam satu ulasan.
- **IDF (Inverse Document Frequency):** seberapa langka kata tersebut di seluruh dokumen.

Kata yang umum seperti "yang" mendapat bobot rendah. Kata yang khas seperti "menyesatkan" atau "recommended" mendapat bobot lebih tinggi.

Data test hanya di-`transform`, tidak di-`fit_transform`, agar vocabulary tetap berasal dari data training dan tidak terjadi data leakage.

### 5.2 Pendekatan 2: Gemini API

Karakteristik eksperimen LLM:

| Aspek | Nilai |
|---|---|
| Model | `gemini-3.1-flash-lite` |
| Teknik prompting | Zero-shot |
| Temperature | 0.0 |
| Max output token | 32 |
| Thinking | Dimatikan untuk model non-lite |
| Normalisasi output | Lowercase, strip whitespace, mapping ke `positif` atau `negatif` |
| Fallback | `tidak_dikenali` jika output gagal diparse |

Prompt berisi:

1. Peran sebagai analis sentimen.
2. Tugas klasifikasi.
3. Definisi label positif dan negatif.
4. Aturan memperhatikan kata negasi.
5. Aturan tie-break jika ragu.
6. Ulasan yang dibungkus tag `<ulasan>` untuk mengurangi risiko prompt injection.
7. Instruksi menjawab hanya satu kata.

LLM dijalankan pada seluruh 40 kalimat unik, bukan hanya 8 kalimat test. Alasannya, LLM tidak membutuhkan training sehingga tidak ada risiko data leakage. Hasilnya bisa dibandingkan langsung dengan cross-validation model klasik pada himpunan kalimat yang sama.

### 5.3 Catatan pergantian model Gemini

Selama pengerjaan, nama model perlu disesuaikan dengan ketersediaan API:

| Model | Status | Keterangan |
|---|---|---|
| `gemini-2.5-flash` | Tidak dipakai | 404, sudah tidak tersedia untuk pengguna baru |
| `gemini-3.5-flash` | Tidak dipakai | 429, kuota key habis saat eksperimen |
| `gemini-3.1-flash-lite` | Dipakai | Berhasil memproses 40 dari 40 kalimat tanpa error |

Nama model dapat diubah di Section 5.2 notebook pada variabel:

```python
MODEL = 'gemini-3.1-flash-lite'
```

---

## 6. Hasil Evaluasi

### 6.1 Test set 8 kalimat

| Pendekatan | Accuracy | Precision macro | Recall macro | F1 macro |
|---|---:|---:|---:|---:|
| Model klasik | 1,00 | 1,00 | 1,00 | 1,00 |
| Gemini 3.1 Flash Lite | 1,00 | 1,00 | 1,00 | 1,00 |

Confusion matrix kedua pendekatan sama:

| | Prediksi positif | Prediksi negatif |
|---|---:|---:|
| Aktual positif | 4 | 0 |
| Aktual negatif | 0 | 4 |

Interpretasi:

- Kedua pendekatan memprediksi semua 8 kalimat test dengan benar.
- Hasil ini seri, tetapi test set hanya 8 kalimat. Satu kesalahan saja akan menggeser accuracy sebesar 12,5%, sehingga hasil test set belum cukup untuk menyimpulkan pemenang.

### 6.2 Seluruh 40 kalimat unik

| Pendekatan | Accuracy | Precision macro | Recall macro | F1 macro | Salah |
|---|---:|---:|---:|---:|---:|
| Model klasik, 5-fold CV | 0,70 | 0,70 | 0,70 | 0,70 | 12 dari 40 |
| Gemini 3.1 Flash Lite | 1,00 | 1,00 | 1,00 | 1,00 | 0 dari 40 |

Confusion matrix model klasik pada 40 kalimat:

| | Prediksi positif | Prediksi negatif |
|---|---:|---:|
| Aktual positif | 14 | 6 |
| Aktual negatif | 6 | 14 |

Confusion matrix Gemini pada 40 kalimat:

| | Prediksi positif | Prediksi negatif |
|---|---:|---:|
| Aktual positif | 20 | 0 |
| Aktual negatif | 0 | 20 |

Interpretasi:

- Model klasik benar pada 28 dari 40 kalimat.
- Kesalahannya seimbang: 6 ulasan positif diprediksi negatif dan 6 ulasan negatif diprediksi positif.
- Untuk use case ini, 6 ulasan negatif yang diprediksi positif atau false negative adalah kesalahan yang paling merugikan karena keluhan pelanggan lolos dari filter.
- Gemini 3.1 Flash Lite benar pada 40 dari 40 kalimat.
- Dataset hanya berisi 40 kalimat yang relatif jelas, sehingga hasil sempurna LLM belum tentu bertahan pada data produksi yang lebih beragam.

### 6.3 Contoh kesalahan model klasik

Model klasik salah pada 12 kalimat. Pola kesalahannya menunjukkan kelemahan bag-of-words dan data training yang kecil.

Contoh ulasan negatif yang diprediksi positif:

- "Pelayanan lambat dan tidak responsif saat dikomplain."
- "Sudah bayar mahal tapi kualitas jauh dari ekspektasi."
- "Bahan murahan, tidak sesuai dengan harga yang dibayarkan."
- "Pengalaman belanja terburuk, tidak akan order lagi disini."

Contoh ulasan positif yang diprediksi negatif:

- "Packing rapi, produk sesuai deskripsi. Recommended seller!"
- "Bahan nyaman dipakai, ukuran pas, tidak mengecewakan."
- "Produk original dan berfungsi dengan baik, terima kasih!"
- "Produknya rapi dan bersih, tidak ada cacat sama sekali."

Penyebab utamanya:

1. Vocabulary hanya dibentuk dari 32 kalimat training.
2. TF-IDF mengabaikan urutan kata dan konteks negasi.
3. Kata seperti "cacat" atau "mengecewakan" mendominasi keputusan, meskipun kalimatnya mengandung negasi.
4. Dengan data kecil, model dapat mempelajari pola kebetulan.

Gemini tidak membuat kesalahan pada 40 kalimat ini, termasuk pada kalimat negasi seperti "tidak ada cacat sama sekali".

### 6.4 File hasil evaluasi

Semua artefak hasil tersedia di folder `documentation/`:

- `comparison_table.csv`: perbandingan pada test set 8 kalimat.
- `comparison_table_40.csv`: perbandingan pada 40 kalimat unik.
- `confusion_matrix_classic_test.csv`: confusion matrix model klasik pada test set.
- `confusion_matrix_llm_test.csv`: confusion matrix LLM pada test set.
- `confusion_matrix_classic_cv_40.csv`: confusion matrix model klasik pada 40 kalimat.
- `confusion_matrix_llm_40.csv`: confusion matrix LLM pada 40 kalimat.
- `model_comparison_summary.png`: grafik ringkasan accuracy dan F1 macro pada 40 kalimat.

---

## 7. Analisis Trade-off dan Limitation

### 7.1 Pendekatan yang lebih unggul dari sisi performa

Perbandingan yang paling dapat dipercaya adalah hasil pada 40 kalimat unik, bukan test set 8 kalimat.

Pada 40 kalimat:

- Model klasik: accuracy 0,70 dan F1 macro 0,70.
- Gemini 3.1 Flash Lite: accuracy 1,00 dan F1 macro 1,00.

**Gemini 3.1 Flash Lite unggul pada eksperimen ini.** Namun kesimpulan ini harus dibaca dalam konteks dataset yang kecil dan relatif mudah. LLM mendapat 1,00 pada 40 kalimat, sementara model klasik hanya dilatih dengan 32 kalimat dan kosakata 131 kata.

### 7.2 Trade-off effort implementasi

| Aspek | Model klasik | LLM API |
|---|---|---|
| Persiapan data | Butuh data berlabel dalam jumlah memadai | Tidak butuh training, tetapi tetap butuh data untuk evaluasi |
| Proses training | Ada, `fit`, perlu tuning dan validasi | Tidak ada |
| Rekayasa utama | Feature engineering: TF-IDF, n-gram, cleaning | Prompt engineering dan normalisasi output |
| Retraining | Wajib bila distribusi bahasa berubah | Tidak wajib, cukup ganti prompt atau model |
| Waktu ke hasil pertama | Lebih lama, preprocessing lalu training lalu evaluasi | Sangat cepat, satu prompt langsung jalan |

LLM jauh lebih murah dari sisi effort awal. Model klasik lebih murah dari sisi biaya berjalan.

### 7.3 Trade-off kecepatan

| Aspek | Model klasik | LLM API |
|---|---|---|
| Latency per prediksi | Milidetik | Ratusan milidetik sampai beberapa detik |
| Batch | Mudah dan cepat | Tergantung rate limit |
| Offline | Bisa | Tidak bisa, butuh internet |
| Skala besar | Sangat cepat | Perlu antrean, retry, dan manajemen kuota |

Pada 40 ulasan, perbedaan latency tidak terasa. Pada 1 juta ulasan, model klasik selesai dalam hitungan menit sampai jam, sedangkan LLM API bisa memerlukan waktu dan biaya yang jauh lebih besar.

### 7.4 Trade-off biaya

| Komponen | Model klasik | LLM API |
|---|---|---|
| Biaya training | Nol sampai kecil, cukup CPU | Tidak ada |
| Biaya per prediksi | Hampir nol setelah model dilatih | Proporsional jumlah token dan volume |
| Skala biaya | Tetap terhadap volume | Linear terhadap volume |
| Biaya tersembunyi | Retraining, maintenance pipeline | Perubahan harga vendor, kuota terlampaui |
| Privasi data | Data tidak keluar dari infrastruktur | Ulasan pelanggan dikirim ke pihak ketiga |

### 7.5 Limitation model klasik

1. Tidak mampu menggeneralisasi dengan baik ke kalimat baru karena vocabulary hanya berasal dari 32 kalimat training.
2. Tidak memahami negasi dengan andal. Contoh: "tidak ada cacat sama sekali" diprediksi negatif.
3. Mengabaikan urutan kata karena TF-IDF memperlakukan teks sebagai kantong kata.
4. Sangat bergantung pada ukuran dan kualitas data. Data kecil membuat model menghafal, bukan menggeneralisasi.
5. Perlu retraining ketika tren bahasa atau jenis produk berubah.
6. Bias kelas mudah muncul jika distribusi label tidak seimbang.

### 7.6 Limitation LLM API

1. Tidak sepenuhnya deterministik meskipun `temperature=0.0`. Output dapat berbeda antar run atau antar versi model.
2. Biaya dan latency tumbuh linear terhadap volume.
3. Rate limit dan ketergantungan vendor. Gangguan penyedia layanan membuat fitur ikut terganggu.
4. Privasi dan kepatuhan. Ulasan pelanggan dikirim ke pihak ketiga.
5. Prompt injection. Pelanggan dapat menulis ulasan berisi instruksi.
6. Kegagalan parsing. Output yang tidak sesuai format harus ditangani eksplisit.
7. Vendor lock-in. Model dapat di-deprecate atau berubah perilakunya.
8. Hasil sempurna pada 40 kalimat belum menjamin performa yang sama pada data produksi.

---

## 8. Rekomendasi Technical Approach

**Rekomendasi:** untuk fase awal, gunakan **Gemini 3.1 Flash Lite dengan prompt zero-shot**. Setelah data berlabel terkumpul cukup banyak, migrasikan ke model klasik dan gunakan LLM sebagai pelabel otomatis atau weak supervision.

### 8.1 Alasan berbasis hasil eksperimen

1. **Dataset yang tersedia belum layak untuk melatih model klasik secara optimal.** Hanya ada 40 kalimat unik, 32 di antaranya untuk training. Model klasik memperoleh accuracy 0,70 dengan 12 kesalahan, sedangkan LLM memperoleh 1,00 tanpa kesalahan pada 40 kalimat yang sama.
2. **Model klasik lemah pada negasi.** Padahal keluhan pelanggan sering berbentuk negasi seperti "tidak berfungsi", "tidak sesuai", dan "belum sampai". LLM menangani negasi tanpa proses training.
3. **Effort implementasi LLM jauh lebih rendah.** Tidak ada training, preprocessing, atau tuning, sehingga cocok untuk eksperimen awal dengan data terbatas dan tenggat pendek.
4. **Volume saat ini masih kecil.** Biaya dan latency API baru menjadi masalah pada skala besar.

### 8.2 Mengapa bukan model klasik sekarang

Bukan karena metodenya buruk, tetapi karena data pendukungnya belum ada. Model klasik tetap lebih ekonomis pada volume besar:

- Biaya per prediksi hampir nol.
- Latency milidetik.
- Data tidak keluar dari infrastruktur.

Kesimpulan ini berubah ketika dataset berisi ribuan ulasan unik dan beragam.

### 8.3 Rencana implementasi bertahap

| Tahap | Aksi | Alasan |
|---|---|---|
| 1. Sekarang | Deploy Gemini 3.1 Flash Lite untuk klasifikasi dan logging hasil ke database | Cepat, tanpa data training |
| 2. 1 sampai 3 bulan | Kumpulkan ulasan produksi dan koreksi manual CS sebagai data berlabel | Membangun dataset yang benar-benar beragam |
| 3. 3 sampai 6 bulan | Latih ulang model klasik dengan data baru dan bandingkan F1 macro terhadap LLM | Model klasik baru layak dievaluasi setelah data cukup |
| 4. Jangka panjang | Pindahkan traffic utama ke model klasik, LLM hanya untuk kasus ambigu atau auto-label | Biaya dan latency turun, privasi lebih baik |

### 8.4 Risiko yang harus dimitigasi

- Privasi data pelanggan.
- Ketergantungan pada satu vendor. Bungkus API di balik interface internal.
- Kegagalan parsing output. Sediakan fallback ke review manual.
- Biaya tak terduga. Pasang batas kuota dan monitoring token.
- Perubahan model atau deprecation. Simpan nama model dan versi di konfigurasi.

---

## 9. Cara Menjalankan

### 9.1 Prasyarat

- Python 3.10 atau lebih baru. Eksperimen ini dijalankan dengan Python 3.14.
- API key Gemini dari `https://aistudio.google.com/app/apikey`.

### 9.2 Instalasi

Dari root project:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Isi `requirements.txt`:

```text
pandas
scikit-learn
google-genai
python-dotenv
jupyter
```

### 9.3 Konfigurasi API key

Salin file example:

```powershell
copy .env.example .env
```

Lalu isi `.env`:

```env
GEMINI_API_KEY=isi_api_key_kamu_di_sini
```

File `.env` sudah masuk `.gitignore` dan tidak akan ter-commit.

### 9.4 Menjalankan notebook

Opsi 1, lewat Jupyter:

```powershell
jupyter lab
```

Lalu buka `notebook/experiment_notebook.ipynb`.

Opsi 2, lewat VS Code:

1. Buka folder project.
2. Buka `notebook/experiment_notebook.ipynb`.
3. Pilih kernel Python yang digunakan.
4. Klik **Run All**.

Opsi 3, lewat terminal:

```powershell
python -m nbconvert --to notebook --execute notebook/experiment_notebook.ipynb --output experiment_notebook.ipynb --ExecutePreprocessor.timeout=600
```

### 9.5 Urutan eksekusi

Jalankan dari atas ke bawah:

1. Import library dan load dataset.
2. EDA dataset.
3. Deduplikasi dan train/test split.
4. TF-IDF.
5. Training Logistic Regression.
6. Prediksi model klasik.
7. 5-fold cross-validation.
8. Setup API Gemini.
9. Definisi prompt dan model.
10. Inference LLM pada 40 kalimat.
11. Evaluasi model klasik.
12. Evaluasi LLM.
13. Tabel perbandingan test set.
14. Tabel perbandingan 40 kalimat dan contoh kesalahan.

Cell inference LLM membutuhkan 40 panggilan API. Waktu eksekusinya sekitar 1 sampai 5 menit tergantung latency dan kuota.

### 9.6 Output yang dihasilkan

Setelah Run All:

- Output metrik dan confusion matrix muncul di notebook.
- `documentation/comparison_table.csv` dibuat atau diperbarui.
- Notebook menyimpan hasil evaluasi pada variabel dan output cell.

### 9.7 Jika API error

| Error | Penyebab | Solusi |
|---|---|---|
| `404 NOT_FOUND` | Model sudah tidak tersedia untuk pengguna baru | Ganti `MODEL` ke model yang tersedia |
| `429 RESOURCE_EXHAUSTED` | Kuota atau rate limit habis | Tunggu reset, ganti key, atau ganti model |
| `403 PERMISSION_DENIED` | API key tidak valid atau API belum aktif | Cek key dan aktifkan Gemini API |
| `NameError: client` | Cell 5.1 belum dijalankan | Jalankan cell setup API |
| `NameError: classify_sentiment_llm` | Cell 5.2 belum dijalankan | Jalankan cell prompt |
| Output LLM kosong | Token output habis untuk thinking | Matikan thinking atau naikkan `max_output_tokens` |

Daftar model yang tersedia untuk key tertentu dapat dicek dengan:

```python
for model in client.models.list():
    print(model.name)
```

---

## 10. Reproducibility

| Komponen | Nilai |
|---|---|
| Random state split | 42 |
| Test size | 0.2 |
| Cross-validation | 5-fold Stratified |
| Model klasik | TF-IDF + Logistic Regression, `max_iter=1000` |
| Model LLM | `gemini-3.1-flash-lite` |
| Temperature | 0.0 |
| Max output token | 32 |
| Teknik prompting | Zero-shot |
| Normalisasi output | Lowercase dan mapping label |

Hasil LLM dapat berbeda sedikit antar run karena sifat model generatif, meskipun temperature 0.0. Simpan nama model dan tanggal eksperimen saat melaporkan hasil.

---

## 11. Checklist Output Assignment

- [x] Experiment notebook berisi kedua pendekatan.
- [x] Dataset asli tersedia di `data/`.
- [x] Evaluation results: accuracy, precision, recall, F1, dan confusion matrix.
- [x] Model comparison table.
- [x] Analisis trade-off dan limitation.
- [x] Technical recommendation.
- [x] `requirements.txt`.
- [x] `README.md`.
- [x] Artefak dokumentasi di folder `documentation/`.
- [x] Nama peserta sudah diisi.
- [x] Repository GitHub sudah memakai format `assignment-model-experiment-Haikal_Satrio`.
- [x] Repository bersifat public dan dapat diakses mentor.
