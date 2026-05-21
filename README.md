# Workflow-CI

Repository ini berisi implementasi **MLflow Project + GitHub Actions CI** untuk melakukan re-training model demand forecasting secara otomatis ketika workflow dipicu.

## Apa isi repo ini

- `MLProject/` berisi project MLflow utama.
- `MLProject/modelling.py` adalah script training demand forecasting per group (Category/ship-state/SKU).
- `MLProject/conda.yaml` adalah environment dependency untuk training.
- `MLProject/MLproject` adalah definisi entry point MLflow.
- `MLProject/namadataset_preprocessing/` berisi dataset input yang dipakai saat training.
- `.github/workflows/ci.yml` adalah workflow GitHub Actions untuk menjalankan MLflow Project.
- `ci_artifacts/` akan berisi artifact hasil CI yang dikomit ke GitHub oleh workflow.

## Alur kerja

Saat ada `push` ke branch `main` atau workflow dijalankan manual, GitHub Actions akan:

1. Menyiapkan Python environment dan memeriksa environment runner.
2. Meng-install dependensi MLflow dan library training.
3. Menjalankan `mlflow run` pada folder `MLProject` (demand forecasting).
4. Mengambil `run_id` MLflow terbaru.
5. Menyalin artifact run ke `ci_artifacts/mlproject/<run_id>/` lalu meng-commit ke GitHub.
6. Membuat Docker image dari model dengan `mlflow models build-docker`.
7. Push image ke Docker Hub.

## Cara pakai lokal

### 1. Masuk ke folder project

```bash
cd MLProject
```

### 2. Siapkan environment

Jika memakai conda:

```bash
conda env create -f conda.yaml
conda activate mlproject-env
```

### 3. Jalankan training manual

```bash
mlflow run . --env-manager local -P data=namadataset_preprocessing/daily_sales_forecasting.csv -P output=artifacts -P target_col=Daily_Revenue

# Contoh demand forecasting per Category
mlflow run . --env-manager local -P data=cleaned_amazon_sales.csv -P output=artifacts -P group_col=Category -P group_value=Kurta -P target_col=Qty
```

Hasil training akan tersimpan di folder `artifacts/` dan terlog ke MLflow run lokal.

## Cara pakai CI di GitHub

### 1. Pastikan workflow ada

File workflow ada di [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

### 2. Tambahkan secrets Docker Hub

Di GitHub repository settings, tambahkan:

- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

### 3. Trigger workflow

Workflow akan jalan otomatis saat ada push ke `main`, atau bisa dijalankan manual dari tab Actions.

## Cara cek berhasil

- Workflow status di GitHub Actions harus sukses.
- Folder `ci_artifacts/mlproject/<run_id>/` harus terisi jika workflow melakukan commit artifact.
- Docker Hub harus punya image dengan nama `DOCKERHUB_USERNAME/mlproject-model`.

## Catatan

Kalau workflow gagal di langkah setup environment, cek file [MLProject/conda.yaml](MLProject/conda.yaml) dan pastikan dependency tidak bentrok.
