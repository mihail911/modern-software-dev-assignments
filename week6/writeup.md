# Week 6 Write-up
Tip: To preview this markdown file
- On Mac, press `Command (⌘) + Shift + V`
- On Windows/Linux, press `Ctrl + Shift + V`

## Instructions

Fill out all of the `TODO`s in this file.

## Submission Details

Name: Rifky Putra Mahardika \
SUNet ID: 2310817210023 \
Citations:
- Warp University (Warp AI support)
- Semgrep Rules Documentation (https://semgrep.dev/explore)
- Gemini AI (Collaborative Debugging and Write-up assistance)
- Cursor AI (Automated code remediation)

This assignment took me about **2** hours to do. 


## Brief findings overview 
> Saya melakukan pemindaian keamanan statis (SAST) menggunakan perintah `semgrep scan --config auto week6`. Karena menggunakan lingkungan Windows, saya harus mengatur variabel environment `$env:PYTHONUTF8 = 1` terlebih dahulu untuk menghindari error encoding karakter saat memuat aturan Semgrep.
>
> Hasil scan menunjukkan **6 temuan kategori "Blocking"** (High Risk). Temuan tersebut meliputi kerentanan kritis seperti SQL Injection, Cross-Site Scripting (XSS), Insecure CORS, dan penggunaan fungsi berbahaya seperti `eval()` serta `shell=True` pada subprocess. Saya memilih untuk memperbaiki 3 isu yang paling berdampak langsung pada keamanan data dan integritas aplikasi: SQL Injection, XSS, dan Wildcard CORS.

## Fix #1
a. File and line(s)
> `week6\backend\app\routers\notes.py` (Baris 71-79)

b. Rule/category Semgrep flagged
> `python.sqlalchemy.security.audit.avoid-sqlalchemy-text`

c. Brief risk description
> Kode menggunakan Python f-string untuk memasukkan input pengguna (`q`) langsung ke dalam statement SQL mentah. Ini adalah celah **SQL Injection** klasik di mana penyerang dapat memanipulasi query untuk mengakses atau menghapus data sensitif di database.

d. Your change (short code diff or explanation, AI coding tool usage)
> Dengan Cursor AI, saya mengubah query menjadi parameterized query. String SQL sekarang menggunakan placeholder `:q` dan nilainya dikirim melalui kamus (dictionary) pada fungsi `execute`, sehingga input user tidak lagi dianggap sebagai bagian dari perintah SQL.

e. Why this mitigates the issue
> Mitigasi ini memastikan input pengguna diperlakukan murni sebagai data literal. Driver database secara otomatis melakukan escaping pada karakter berbahaya, sehingga perintah SQL penyerang tidak akan pernah dieksekusi.

## Fix #2
a. File and line(s)
> `week6\frontend\app.js` (Baris 14)

b. Rule/category Semgrep flagged
> `javascript.browser.security.insecure-document-method`

c. Brief risk description
> Penggunaan property `innerHTML` untuk menampilkan data dari input pengguna (`n.title` dan `n.content`) memicu risiko **Stored XSS (Cross-Site Scripting)**. Penyerang bisa menyisipkan skrip jahat ke database yang akan tereksekusi secara otomatis di browser pengguna lain.

d. Your change (short code diff or explanation, AI coding tool usage)
> Saya menggunakan **Cursor AI** untuk mengganti `innerHTML` dengan metode yang lebih aman. Logika pembuatan elemen daftar (`li`) diperbaiki agar menggunakan `textContent` untuk mengisi data teks.

e. Why this mitigates the issue
> `textContent` secara otomatis melakukan encoding pada karakter khusus (seperti `<` dan `>`). Browser akan menampilkan karakter tersebut sebagai teks biasa dan tidak akan merender atau mengeksekusi tag HTML/script apa pun yang ada di dalamnya.

## Fix #3
a. File and line(s)
> `week6\backend\app\main.py` (Baris 24)

b. Rule/category Semgrep flagged
> `python.fastapi.security.wildcard-cors.wildcard-cors`

c. Brief risk description
> Konfigurasi CORS menggunakan wildcard `allow_origins=["*"]`. Ini sangat tidak aman karena membiarkan domain mana pun di internet melakukan permintaan API ke backend ini, yang meningkatkan risiko serangan *Cross-Site Request Forgery* (CSRF).

d. Your change (short code diff or explanation, AI coding tool usage)
> Menggunakan **Cursor AI**, saya membatasi izin akses hanya untuk origin lokal yang tepercaya, yaitu `http://localhost:3000` dan `http://127.0.0.1:3000`.

e. Why this mitigates the issue
> Dengan membatasi origin, browser akan memblokir semua permintaan dari domain yang tidak dikenal, memastikan hanya frontend resmi kita yang diizinkan untuk berkomunikasi dengan API backend.