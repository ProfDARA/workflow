# Workflow-CI

Repository ini berisi implementasi **MLflow Project + GitHub Actions CI** untuk melakukan re-training model secara otomatis ketika workflow dipicu.

## Apa isi repo ini

- `MLProject/` berisi project MLflow utama.
- `MLProject/modelling.py` adalah script training model.
- `MLProject/conda.yaml` adalah environment dependency untuk training.
- `MLProject/MLproject` adalah definisi entry point MLflow.
- `MLProject/namadataset_preprocessing/` berisi dataset input yang dipakai saat training.
- `.github/workflows/ci.yml` adalah workflow GitHub Actions untuk menjalankan MLflow Project.
- `ci_artifacts/` akan berisi artifact hasil CI yang dikomit oleh workflow.

## Alur kerja

Saat ada `push` ke branch `main` atau workflow dijalankan manual, GitHub Actions akan:

1. Menyiapkan environment conda dari `MLProject/conda.yaml`.
2. Menjalankan `mlflow run` pada folder `MLProject`.
3. Menyimpan artifact hasil training.
4. Mengambil run MLflow terbaru.
5. Membuat Docker image dari model dengan `mlflow models build-docker`.
6. Push image ke Docker Hub.

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
mlflow run . --env-manager local -P data=namadataset_preprocessing/daily_sales_forecasting.csv -P output=artifacts
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
- Artifact `mlproject-artifacts` harus muncul di hasil run.
- Folder `ci_artifacts/mlproject/` harus terisi jika workflow melakukan commit artifact.
- Docker Hub harus punya image dengan nama `DOCKERHUB_USERNAME/mlproject-model`.

## Catatan

Kalau workflow gagal di langkah setup environment, cek file [MLProject/conda.yaml](MLProject/conda.yaml) dan pastikan dependency tidak bentrok.
