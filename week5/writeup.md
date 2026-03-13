# Week 5 Write-up
Tip: To preview this markdown file
- On Mac, press `Command (⌘) + Shift + V`
- On Windows/Linux, press `Ctrl + Shift + V`

## INSTRUCTIONS

Fill out all of the `TODO`s in this file.

## SUBMISSION DETAILS

Name: Rifky Putra Mahardika \
SUNet ID: 2310817210023 \
Citations: 
- Warp University (warp.dev/university)
- Agentic Development Patterns (Course Material)
- Gemini AI (Collaborative Debugging and Logic Verification)

This assignment took me about **2** hours to do. 


## YOUR RESPONSES
### Automation A: Warp Drive "Agentic-Health-Check" Workflow

a. Design of each automation, including goals, inputs/outputs, steps
- **Goal**: Menstandarisasi proses verifikasi kode (formatting, linting, dan testing) menjadi satu perintah yang bisa dipanggil cepat di Windows.
- **Inputs**: Definisi Workflow di Warp Drive dengan sintaks PowerShell.
- **Outputs**: Laporan status eksekusi Black, Ruff, dan Pytest yang terkonsolidasi.
- **Steps**: Menjalankan `black .`, dilanjutkan `ruff check . --fix`, dan diakhiri dengan `python -m pytest`.

b. Before vs. after (i.e. manual workflow vs. automated workflow)
- **Manual**: Saya harus mengetik tiga perintah panjang satu per satu secara manual. Seringkali saya lupa menjalankan tes setelah merapikan kode, sehingga ada bug yang tidak terdeteksi.
- **Automated**: Cukup dengan satu klik atau mengetik `Agentic-Health-Check` di Warp, seluruh rangkaian pengecekan berjalan otomatis.

c. Autonomy levels used for each completed task (what code permissions, why, and how you supervised)
- **Level**: Medium Autonomy. AI mengeksekusi urutan perintah sistem yang sudah ditentukan. Saya mengawasi jalannya proses untuk memastikan semua indikator berwarna "Hijau".

d. (if applicable) Multi‑agent notes: roles, coordination strategy, and concurrency wins/risks/failures
> TODO

e. How you used the automation (what pain point it resolves or accelerates)
- Saya menjalankan workflow ini setiap kali agen AI selesai mengerjakan sebuah fitur (seperti fitur Delete atau Bulk Complete) untuk memastikan tidak ada kerusakan pada sistem.



### Automation B: Multi-Agent Concurrent Session

a. Design of each automation, including goals, inputs/outputs, steps
- **Goal**: Mempercepat pengembangan aplikasi dengan mengerjakan fitur Backend (API) dan Frontend (UI) secara bersamaan menggunakan agen AI yang berbeda.
- **Inputs**: Prompt paralel di panel Warp yang terpisah dengan referensi "@TASKS.md".
- **Outputs**: Implementasi fitur Delete Note (Task 3) dan Action Items Bulk-Complete (Task 4).
- **Steps**: Membagi terminal menjadi dua panel (split pane). Panel 1 (Agen 1) fokus pada logika Backend Notes. Panel 2 (Agen 2) fokus pada API Bulk-Complete dan tombol UI.

b. Before vs. after (i.e. manual workflow vs. automated workflow)
- **Manual**: Saya harus menyelesaikan kode backend terlebih dahulu, mengetesnya, baru kemudian pindah ke frontend—proses yang linear dan lambat.
- **Automated**: Kedua agen bekerja secara paralel. Saat Agen 1 menulis logika database, Agen 2 sudah membangun UI-nya. Ini memungkinkan penyelesaian dua tugas kompleks dalam satu siklus kerja.

c. Autonomy levels used for each completed task (what code permissions, why, and how you supervised)
- **Level**: High Autonomy. Saya memberikan izin kepada agen untuk mengubah logika inti, model database, dan file UI. Saya bertindak sebagai "Arsitek" yang memberikan arahan dan melakukan review final.

d. (if applicable) Multi‑agent notes: roles, coordination strategy, and concurrency wins/risks/failures
- **Roles**: Agen 1 sebagai Spesialis Backend (CRUD), Agen 2 sebagai Spesialis Full-stack (Bulk API & UI).
- **Coordination**: Kedua agen diinstruksikan membaca "@TASKS.md" agar tetap selaras.
- **Concurrency Wins**: Menghemat waktu pengembangan hingga 40% karena eksekusi paralel.
- **Risks/Failures**: Sempat terjadi tabrakan kode (collision) di "schemas.py". Saya menyelesaikannya dengan menggabungkan perubahan dari kedua agen secara manual sebelum melakukan "Apply".

e. How you used the automation (what pain point it resolves or accelerates)
- Menghilangkan hambatan pengembangan linear. Dengan setup ini, saya bisa langsung memperbaiki error akses file ("PermissionError") di satu panel sambil tetap membiarkan agen lain melanjutkan pengembangan UI di panel sebelahnya.


### (Optional) Automation C: Any Additional Automations
a. Design of each automation, including goals, inputs/outputs, steps
> TODO

b. Before vs. after (i.e. manual workflow vs. automated workflow)
> TODO

c. Autonomy levels used for each completed task (what code permissions, why, and how you supervised)
> TODO

d. (if applicable) Multi‑agent notes: roles, coordination strategy, and concurrency wins/risks/failures
> TODO

e. How you used the automation (what pain point it resolves or accelerates)
> TODO

