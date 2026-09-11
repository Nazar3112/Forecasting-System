# 📘 Panduan Penggunaan & Dokumentasi Sistem Prediksi Tren Penjualan Retail
### Sistem Peramalan Berbasis Kategori Produk Menggunakan Model Hibrida Facebook Prophet + LightGBM
**Studi Kasus:** PT. Indomarco Prismatama (Indomaret) · DC Cirebon  
**Pengembang:** Alfiyan Nazar (220511053) — Teknik Informatika, Universitas Muhammadiyah Cirebon (2026)

---

## 📌 Daftar Isi
1. [Pengantar & Konsep Dasar (Untuk Orang Awam)](#1-pengantar--konsep-dasar-untuk-orang-awam)
2. [Arsitektur & Keunggulan Sistem](#2-arsitektur--keunggulan-sistem)
3. [Format & Persiapan Data Transaksi (CSV)](#3-format--persiapan-data-transaksi-csv)
4. [Langkah demi Langkah Penggunaan Aplikasi](#4-langkah-demi-langkah-penggunaan-aplikasi)
5. [Penjelasan Rinci Fitur & Menu Dashboard](#5-penjelasan-rinci-fitur--menu-dashboard)
   - [A. Sidebar (Panel Konfigurasi)](#a-sidebar-panel-konfigurasi)
   - [B. Tab 1: Overview (Ringkasan Data)](#b-tab-1-overview-ringkasan-data)
   - [C. Tab 2: Kurva Peramalan (Visualisasi Tren)](#c-tab-2-kurva-peramalan-visualisasi-tren)
   - [D. Tab 3: Evaluasi Akurasi (Validasi Kinerja Model)](#d-tab-3-evaluasi-akurasi-validasi-kinerja-model)
   - [E. Tab 4: Ekspor Data (Laporan Proyeksi)](#e-tab-4-ekspor-data-laporan-proyeksi)
6. [Memahami Istilah & Metrik Evaluasi](#6-memahami-istilah--metrik-evaluasi)
7. [Panduan Presentasi & Demo Sistem (Bahan Sidang / Rapat)](#7-panduan-presentasi--demo-sistem-bahan-sidang--rapat)
8. [Tanya Jawab Umum (FAQ)](#8-tanya-jawab-umum-faq)

---

## 1. Pengantar & Konsep Dasar (Untuk Orang Awam)

### Apa Masalah yang Ingin Diselesaikan?
Di bisnis ritel modern seperti Indomaret, menentukan jumlah stok barang yang harus dikirim dari Distribution Center (DC) ke toko-toko adalah tantangan besar:
- **Kelebihan Stok (Overstock):** Modal tertahan di gudang, risiko barang rusak atau kedaluwarsa meningkat, serta memakan ruang simpan.
- **Kekurangan Stok (Stockout):** Toko kehabisan barang saat konsumen butuh, berakibat hilangnya potensi penjualan dan penurunan kepuasan pelanggan.

### Bagaimana Sistem Ini Menjawab Masalah Tersebut?
Sistem ini memprediksi **berapa banyak unit barang yang akan terjual di masa depan** untuk setiap kelompok/divisi produk. Dengan prediksi ini, tim pengadaan dan logistik dapat merencanakan stok secara akurat dan tepat waktu.

### Apa Itu Model "Hibrida Prophet + LightGBM"?
Alih-alih hanya mengandalkan satu rumus atau satu algoritma, sistem ini menggabungkan dua kekuatan terbaik:
1. **Facebook Prophet (Model Statistik & Deret Waktu):**
   - Bertugas menangkap **pola makro**: tren jangka panjang (apakah penjualan cenderung naik atau turun) dan pola musiman (apakah ada kenaikan rutin di awal bulan atau hari libur).
2. **LightGBM (Model Machine Learning Berbasis Pohon Keputusan):**
   - Bertugas sebagai **korektor cerdas**: menangkap sisa kesalahan (*residual error*) yang meleset dari prediksi Prophet. LightGBM mempelajari pola kompleks, hubungan antar-minggu (*lag features*), dan fluktuasi mendadak.
3. **Hasil Hibrida:**
   $$\text{Prediksi Akhir} = \text{Estimasi Prophet} + \text{Koreksi Residual LightGBM}$$
   Pendekatan gabungan ini terbukti menghasilkan tingkat kesalahan (galat) yang jauh lebih kecil dibanding model konvensional tunggal.

---

## 2. Arsitektur & Keunggulan Sistem

| Fitur / Karakteristik | Penjelasan | Keuntungan |
| :--- | :--- | :--- |
| **Stateless In-Memory** | Tidak menyimpan data ke database server atau cloud permanen. Semua pemrosesan data berlangsung di memori RAM selama sesi aktif. | **Privasi & Keamanan Tinggi:** Data log penjualan perusahaan tidak bocor atau tersimpan di pihak ketiga. |
| **Desain Finansial Modern (Binance Theme)** | Tampilan bertema gelap (*dark canvas* `#0b0e11`) dengan aksen kuning khas (*Binance Yellow* `#fcd535`) dan indikator arah trading hijau/merah. | Nyaman dipandang dalam waktu lama, kontras tinggi, data numerik mudah dibaca sekilas. |
| **Pembersihan Data Otomatis** | Otomatis menyaring transaksi retur/batal, menghapus nilai negatif/kosong, dan melakukan agregasi tanggal. | Pengguna tidak perlu repot membersihkan Excel secara manual. |
| **Ekspor Langsung ke CSV** | Menghasilkan berkas CSV standar ber-encode UTF-8 lengkap dengan catatan metadata resmi. | Langsung kompatibel dengan Microsoft Excel, Google Sheets, atau sistem ERP internal. |

---

## 3. Format & Persiapan Data Transaksi (CSV)

Sistem menerima berkas log transaksi kasir kas POS (`.csv`). Berkas harus memuat kolom-kolom berikut (nama kolom menggunakan huruf kapital):

| Nama Kolom | Contoh Nilai | Keterangan |
| :--- | :--- | :--- |
| **`TANGGAL`** | `01-01-2024` atau `2024-01-01` | Tanggal terjadinya transaksi penjualan di kasir. |
| **`RTYPE`** | `'J'` | Tipe rekaman. Sistem hanya mengambil kode `'J'` (*Jual riil*), otomatis mengabaikan retur/batal. |
| **`DIV`** | `'01'`, `'05'` | Kode Divisi produk (kelompok besar, biasanya terdiri dari ~55 kelompok). |
| **`CAT_COD`** | `'010101'` | Kode Sub-Kategori produk (tingkat spesifik, ~289 kode). |
| **`QTY`** | `2`, `10` | Jumlah kuantitas unit barang yang terjual (harus angka positif > 0). |

> 💡 **Catatan:** Sistem secara otomatis menangani tanda kutip tunggal (`'01'`), tanda spasi ekstra, format tanggal ganda, dan membuang baris yang rusak secara otomatis tanpa menghentikan aplikasi.

---

## 4. Langkah demi Langkah Penggunaan Aplikasi

Alur kerja peramalan dirancang sangat sederhana dalam **5 langkah mudah**:

```
[1. Unggah CSV] ➔ [2. Pilih Pengelompokan & Interval] ➔ [3. Pilih Kategori] ➔ [4. Atur Horizon] ➔ [5. Jalankan Peramalan]
```

1. **Jalankan Aplikasi**  
   Buka terminal/PowerShell di folder proyek, ketik:
   ```bash
   streamlit run app.py
   ```
   Aplikasi otomatis terbuka di peramban web pada alamat `http://localhost:8501`.
2. **Unggah File CSV** pada panel kiri (Sidebar).
3. **Pilih Pengelompokan:** Gunakan `DIV` untuk melihat tren divisi besar, atau `CAT_COD` untuk melihat kategori spesifik.
4. **Pilih Interval Waktu:** Pilih `Mingguan (W-MON)` untuk perencanaan operasional gudang, atau `Bulanan (MS)` untuk perencanaan manajerial.
5. **Pilih Kategori Produk:** Pilih kode kategori yang ingin diproyeksikan dari menu dropdown.
6. **Atur Horizon Proyeksi:** Geser slider untuk menentukan berapa minggu/bulan ke depan yang ingin diprediksi (misal: 12 periode).
7. **Klik Tombol Kuning "Jalankan Peramalan"**: Tunggu beberapa detik, hasil analisis dan grafik interaktif akan langsung tersaji di layar utama.

---

## 5. Penjelasan Rinci Fitur & Menu Dashboard

### A. Sidebar (Panel Konfigurasi)
Panel di sisi kiri layar merupakan pusat kendali aplikasi:

1. **Brand Mark & Identitas:** Menampilkan judul sistem dan nama peneliti skripsi.
2. **Bagian 1 — Data Transaksi:** Area *drag-and-drop* untuk mengunggah file `.csv`.
3. **Bagian 2 — Konfigurasi:**
   - **Pengelompokan:** Memilih tingkat agregasi data (`DIV` vs `CAT_COD`).
   - **Interval Waktu:** Menentukan frekuensi deret waktu (Mingguan / Bulanan).
4. **Bagian 3 — Kategori Produk:** Menampilkan daftar kategori yang memenuhi syarat kelayakan data (memiliki $\ge 30$ titik data agar model dapat belajar secara akurat).
5. **Bagian 4 — Parameter Model:**
   - **Dekomposisi Musiman:**
     - *Aditif:* Dipilih jika fluktuasi musiman relatif stabil dari waktu ke waktu.
     - *Multiplikatif:* Dipilih jika ayunan musiman membesar seiring pertumbuhan penjualan.
   - **Horizon Proyeksi:** Menentukan jumlah periode masa depan yang akan dihitung (1 s/d 52 periode).
6. **Tombol "Jalankan Peramalan":** Tombol aksi utama dengan visual kuning Binance yang menyala saat parameter siap.

---

### B. Tab 1: Overview (Ringkasan Data)
Tab pertama berfungsi untuk memahami gambaran umum dataset sebelum meramal:

- **Kartu Ringkasan Berkas:** Menampilkan nama berkas yang sedang aktif, rentang tanggal transaksi historis, total baris bersih yang valid, dan jumlah kategori aktif.
- **Grid 4 Kartu Statistik:**
  1. *Total Transaksi:* Jumlah baris log kasir valid yang berhasil diproses.
  2. *Kategori Aktif:* Jumlah divisi/kategori yang memenuhi syarat kelayakan data.
  3. *Total Kuantitas:* Akumulasi total unit fisik barang yang terjual dalam dataset.
  4. *Periode Data:* Rentang waktu dari transaksi pertama hingga transaksi terakhir.
- **Tabel Deret Waktu (Top 5 Kategori Teratas):** Cuplikan data 100 baris pertama dari 5 kategori dengan volume transaksi tertinggi untuk memeriksa struktur data mentah yang telah diagregasi.

---

### C. Tab 2: Kurva Peramalan (Visualisasi Tren)
Tab kedua menyajikan visualisasi grafis inti dari hasil peramalan:

- **4 Kartu Metrik KPI (Bagian Atas):**
  1. *Penjualan Historis:* Total kuantitas barang yang pernah terjual di kategori terpilih, dilengkapi persentase kenaikan/penurunan tren terkini.
  2. *MAPE Hibrida:* Persentase rata-rata kesalahan prediksi model hibrida pada data uji (dilengkapi indikator kelayakan: Sangat Baik / Baik / Layak).
  3. *RMSE:* Nilai akar rata-rata kuadrat galat (dalam satuan unit fisik barang).
  4. *MAE:* Rata-rata selisih absolut antara penjualan riil dan prediksi per periode.

- **Grafik Garis Interaktif Plotly:**
  - ⚪ **Garis Abu-abu Terang:** Data penjualan aktual historis yang sesungguhnya terjadi di masa lalu.
  - 🔘 **Garis Abu-abu Titik-titik:** Hasil estimasi model Prophet tunggal pada data uji (20% periode terakhir).
  - 🟢 **Garis Hijau Solid (*Trading Green*):** Hasil prediksi model gabungan Hibrida (Prophet + LightGBM) pada data uji.
  - 🟡 **Garis Kuning Putus-putus (*Binance Yellow*):** **Proyeksi penjualan masa depan** untuk periode yang belum terjadi.
  - 🟨 **Arsiran Zona Uji:** Menandai area 20% data terakhir tempat model diuji akurasinya dengan data aktual yang disembunyikan selama pelatihan.
  - 📍 **Garis Vertikal "Mulai Proyeksi":** Batas tegas pemisah antara data masa lalu dan proyeksi masa depan.

> 🔍 **Fitur Interaktivitas Grafik:**
> - Arahkan kursor (*hover*) ke titik manapun pada grafik untuk melihat rincian tanggal dan angka unit secara presisi.
> - Klik pada nama trace di legenda untuk menyembunyikan/menampilkan kurva tertentu (misal: mematikan garis Prophet tunggal agar grafik lebih bersih).
> - Gunakan seleksi klik-tahan untuk memperbesar (*zoom*) bagian grafik tertentu, dan klik ganda untuk kembali ke tampilan awal.

---

### D. Tab 3: Evaluasi Akurasi (Validasi Kinerja Model)
Tab ketiga dirancang khusus untuk kebutuhan akademis, pengujian reliabilitas, dan pembuktian performa algoritma:

1. **Kartu Status Kualitas Model:**
   - Menyajikan interpretasi MAPE secara langsung berdasarkan kriteria resmi industri.
   - Dilengkapi garis batas warna semantik:
     - 🟢 **Hijau:** Sangat Baik ($\text{MAPE} < 10\%$)
     - 🟡 **Kuning:** Baik ($10\% \le \text{MAPE} < 20\%$)
     - 🔴 **Merah:** Layak / Perlu Perhatian ($\text{MAPE} \ge 20\%$)
2. **Tabel Komparasi Model (Prophet vs Hibrida):**
   - Menjajarkan metrik MAE, MAPE, dan RMSE antara **Prophet Mandiri** vs **Model Hibrida**.
   - Menunjukkan bukti empiris apakah penambahan komponen LightGBM berhasil memangkas galat prediksi.
3. **Grafik Analisis Galat Residual:**
   - Menampilkan diagram batang yang membandingkan besarnya galat absolut Prophet vs Hibrida di tiap periode data uji.
   - Membuktikan periode mana saja di mana LightGBM berhasil memperbaiki kesalahan Prophet.
4. **Grafik Pentingnya Fitur (*Feature Importance*) LightGBM:**
   - Memperlihatkan variabel apa saja yang paling berkontribusi dalam koreksi residual (misal: penjualan 1 minggu lalu/`lag_1`, penjualan 2 minggu lalu/`lag_2`, indeks minggu, atau efek kalender).

---

### E. Tab 4: Ekspor Data (Laporan Proyeksi)
Tab keempat digunakan untuk mengonversi hasil ramalan ke dalam dokumen operasional:

1. **Tabel Ringkasan Proyeksi:**
   - Menyajikan tabel dua kolom: Tanggal Periode Masa Depan dan Estimasi Jumlah Unit yang harus disediakan (dibulatkan ke bilangan bulat terdekat).
2. **Tombol Unduh CSV (*Download Button*):**
   - Mengunduh berkas dengan penamaan otomatis `proyeksi_[KATEGORI].csv`.
   - Berkas menyertakan metadata resmi di baris pembuka:
     ```csv
     # Laporan Proyeksi Penjualan Retail
     # Kategori/Divisi : 01
     # Model           : Hibrida Prophet + LightGBM
     # Dibuat oleh     : Sistem Prediksi Tren Penjualan - Indomarco Prismatama
     #
     Periode,Proyeksi_Penjualan_Unit,Estimasi_Prophet_Unit,Koreksi_LightGBM_Unit
     08-01-2024,4250,4100,150
     ...
     ```
3. **Tabel Detail Hasil Uji:**
   - Menyajikan tabel komparasi lengkap data uji: Tanggal, Nilai Aktual ($y$), Prediksi Prophet ($y_{\text{prophet}}$), Prediksi Hibrida ($y_{\text{hybrid}}$), dan Nilai Residual ($e$).

---

## 6. Memahami Istilah & Metrik Evaluasi

Untuk mempermudah penjelasan kepada penguji atau pengguna non-teknis:

| Istilah | Kepanjangan / Definisi | Cara Membacanya untuk Orang Awam |
| :--- | :--- | :--- |
| **MAPE** | *Mean Absolute Percentage Error* | Rata-rata persentase melesetnya ramalan. Jika MAPE = $5\%$, artinya akurasi prediksi rata-rata adalah $95\%$. |
| **MAE** | *Mean Absolute Error* | Rata-rata jumlah unit barang yang meleset per periode tanpa memandang arah positif/negatif. |
| **RMSE** | *Root Mean Squared Error* | Mirip MAE, namun memberikan penalti lebih berat pada kesalahan ramalan yang bernilai sangat besar/ekstrem. |
| **Horizon** | Jangka Waktu Peramalan | Seberapa jauh ke masa depan kita ingin meramal (misal: 8 minggu ke depan). |
| **Residual** | Selisih Galat ($y - \hat{y}$) | Angka yang gagal diprediksi oleh model Prophet, kemudian diserahkan ke LightGBM untuk dipelajari polanya. |

### Standar Kriteria Akurasi MAPE (Lewis, 1982):
- **< 10%** : Kemampuan peramalan **Sangat Baik** (*Highly Accurate*).
- **10% – 20%** : Kemampuan peramalan **Baik** (*Good Forecast*).
- **20% – 50%** : Kemampuan peramalan **Layak / Wajar** (*Reasonable Forecast*).
- **> 50%** : Kemampuan peramalan **Kurang Akurat** (*Inaccurate*).

---

## 7. Panduan Presentasi & Demo Sistem (Bahan Sidang / Rapat)

Jika Anda ingin mendemonstrasikan aplikasi ini di depan **Dosen Penguji** atau **Pimpinan Manajemen**, ikuti alur narasi 5-7 menit berikut:

### 1. Pembukaan (30 Detik)
> *"Selamat pagi/siang Bapak/Ibu. Hari ini saya mendemonstrasikan Sistem Prediksi Tren Penjualan Retail PT. Indomarco Prismatama. Sistem ini dirancang untuk mengatasi masalah kelebihan dan kekurangan stok di Distribution Center dengan memanfaatkan pendekatan mutakhir: Model Hibrida Facebook Prophet dan LightGBM."*

### 2. Penjelasan Arsitektur & Keamanan (1 Menit)
> *"Aplikasi ini dibangun berbasis web interaktif dengan arsitektur Stateless In-Memory. Artinya, seluruh data transaksi sensitif perusahaan hanya diproses langsung di RAM dan tidak disimpan ke server luar atau database permanen, sehingga privasi data transaksi terjamin 100%."*

### 3. Demo Langkah 1 & 2: Unggah & Eksplorasi Data (1 Menit)
> *"Pertama, saya mengunggah berkas log transaksi kasir kas POS. Di Tab Overview, sistem secara instan memvalidasi integritas data: menyaring data batal/retur, menampilkan total transaksi bersih, rentang tanggal historis, serta volume penjualan 5 kategori teratas."*

### 4. Demo Langkah 3 & 4: Peramalan & Nilai Tambah Model Hibrida (2 Menit)
> *"Sekarang saya memilih salah satu divisi produk, mengatur interval waktu mingguan, dan horizon proyeksi 12 minggu ke depan, lalu klik 'Jalankan Peramalan'.*  
> *Pada Tab Kurva Peramalan, perhatikan kurva grafik: garis abu-abu adalah penjualan riil, garis titik-titik adalah estimasi Prophet, dan garis hijau adalah hasil model Hibrida.*  
> *Di sini terlihat jelas bahwa model Hibrida mampu mengikuti lekukan data aktual secara jauh lebih presisi dibandingkan model Prophet mandiri. Garis kuning putus-putus menunjukkan rekomendasi kuantitas yang harus disiapkan gudang untuk 12 minggu mendatang."*

### 5. Bukti Ilmiah di Tab Evaluasi & Ekspor (1.5 Menit)
> *"Pada Tab Evaluasi Akurasi, kita membuktikan keunggulan algoritma secara empiris melalui tabel komparasi: terlihat nilai MAPE berkurang secara signifikan berkat koreksi residual oleh LightGBM. Terakhir, pada Tab Ekspor Data, tim logistik dapat langsung mengunduh hasil proyeksi dalam format CSV siap pakai untuk dimasukkan ke sistem pengadaan."*

### 6. Penutup (30 Detik)
> *"Dengan sistem ini, proses perencanaan logistik Indomarco yang tadinya manual dan reaktif kini dapat beralih menjadi berbasis data prediktif dan proaktif. Terima kasih, saya persilakan jika ada pertanyaan."*

---

## 8. Tanya Jawab Umum (FAQ)

**Q: Mengapa memilih tema gelap bernuansa Binance?**  
**A:** Desain terinspirasi dari standar aplikasi finansial profesional (Binance Design System). Tema gelap dengan kontras aksen kuning memudahkan pengambil keputusan menganalisis grafik angka numerik tebal dalam waktu lama tanpa kelelahan mata, dengan pemisahan warna semantik trading (hijau untuk naik/baik, merah untuk turun/evaluasi).

**Q: Mengapa tidak semua kategori muncul di daftar pilihan?**  
**A:** Sistem menerapkan ambang batas ilmiah minimal $\ge 30$ titik observasi historis. Kategori dengan riwayat data terlalu sedikit sengaja tidak ditampilkan untuk mencegah model menghasilkan prediksi yang *overfitting* atau tidak dapat dipertanggungjawabkan secara statistik.

**Q: Apakah data saya aman saat diunggah ke aplikasi ini?**  
**A:** Sangat aman. Sistem bersifat *stateless* (tidak memiliki database dan tidak menyimpan berkas ke *hard drive* server). Ketika sesi peramban ditutup atau di-refresh, semua data di memori akan langsung terhapus.

**Q: Apa yang harus dilakukan jika nilai MAPE di atas 20%?**  
**A:** Hal tersebut wajar terjadi pada kategori barang musiman ekstrem atau barang yang penjualannya jarang (*intermittent demand*). Anda dapat mencoba mengubah interval waktu ke Bulanan (`MS`) atau mengubah dekomposisi musiman menjadi `Multiplikatif` pada panel konfigurasi.

---
*Dokumentasi ini disusun sebagai pelengkap materi Tugas Akhir / Skripsi S1 Teknik Informatika UMC (2026).*
