import os
import sqlite3
from pathlib import Path

import pandas as pd

# Paths relative to this script
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "cdss.db"

# Dataset paths
DDINTER_DIR = BASE_DIR.parent / "ddinter"
ICD10_DIR = BASE_DIR.parent / "icd102019enMeta"

def init_db(conn):
    """Membuat skema tabel untuk ICD-10 dan DDInter."""
    cursor = conn.cursor()
    
    # Tabel ICD-10 Chapters
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS icd10_chapters (
            chapter_id TEXT PRIMARY KEY,
            chapter_name TEXT NOT NULL
        )
    """)
    
    # Tabel ICD-10 Groups
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS icd10_groups (
            group_id TEXT PRIMARY KEY,
            end_code TEXT NOT NULL,
            chapter_id TEXT NOT NULL,
            group_name TEXT NOT NULL,
            FOREIGN KEY(chapter_id) REFERENCES icd10_chapters(chapter_id)
        )
    """)
    
    # Tabel ICD-10 Codes
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS icd10_codes (
            code TEXT PRIMARY KEY,
            description TEXT NOT NULL,
            group_id TEXT,
            chapter_id TEXT,
            FOREIGN KEY(group_id) REFERENCES icd10_groups(group_id),
            FOREIGN KEY(chapter_id) REFERENCES icd10_chapters(chapter_id)
        )
    """)
    
    # Tabel Interaksi Obat
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS drug_interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            drug_a TEXT NOT NULL,
            drug_b TEXT NOT NULL,
            level TEXT NOT NULL
        )
    """)
    
    # Indeks untuk mempercepat pencarian
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_drug_a ON drug_interactions(drug_a)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_drug_b ON drug_interactions(drug_b)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_icd10_group ON icd10_codes(group_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_icd10_chapter ON icd10_codes(chapter_id)")
    
    conn.commit()

def load_ddinter(conn):
    """Membaca file CSV DDInter dan menyimpannya ke database."""
    print("Memproses dataset DDInter...")
    
    # Ambil semua file CSV DDInter di direktori
    csv_files = list(DDINTER_DIR.glob("ddinter_downloads_code_*.csv"))
    
    if not csv_files:
        print(f"Tidak ada file CSV ditemukan di: {DDINTER_DIR}")
        return
        
    total_imported = 0
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            
            # Cek apakah kolom yang diperlukan ada
            if not all(col in df.columns for col in ['Drug_A', 'Drug_B', 'Level']):
                print(f"Melewati {file_path.name}: kolom Drug_A, Drug_B, atau Level tidak ditemukan.")
                continue
                
            # Ambil kolom yang diperlukan
            df_clean = df[['Drug_A', 'Drug_B', 'Level']]
            
            # Masukkan ke tabel drug_interactions
            df_clean.to_sql('drug_interactions', conn, if_exists='append', index=False, 
                            chunksize=1000, method='multi')
            print(f"Berhasil mengimpor {len(df_clean)} interaksi obat dari {file_path.name}")
            total_imported += len(df_clean)
            
        except Exception as e:
            print(f"Gagal memproses file {file_path.name}: {e}")
            
    print(f"Selesai! Total keseluruhan interaksi obat yang diimpor: {total_imported}")

def load_icd10(conn):
    """Membaca file txt ICD-10 dan menyimpannya ke database dengan relasi Chapter dan Group."""
    print("Memproses dataset ICD-10 (Chapters, Groups, dan Codes)...")
    
    # 1. Load Chapters
    file_chapters = ICD10_DIR / "icd102019syst_chapters.txt"
    if file_chapters.exists():
        df_chap = pd.read_csv(file_chapters, sep=';', header=None, encoding='utf-8', on_bad_lines='skip')
        df_chap_clean = pd.DataFrame({
            'chapter_id': df_chap[0].astype(str).str.strip(),
            'chapter_name': df_chap[1].astype(str).str.strip()
        }).drop_duplicates(subset=['chapter_id'])
        df_chap_clean.to_sql('icd10_chapters', conn, if_exists='append', index=False)
        print(f"Berhasil mengimpor {len(df_chap_clean)} chapters ICD-10")
    
    # 2. Load Groups
    file_groups = ICD10_DIR / "icd102019syst_groups.txt"
    if file_groups.exists():
        df_group = pd.read_csv(file_groups, sep=';', header=None, encoding='utf-8', on_bad_lines='skip')
        df_group_clean = pd.DataFrame({
            'group_id': df_group[0].astype(str).str.strip(),
            'end_code': df_group[1].astype(str).str.strip(),
            'chapter_id': df_group[2].astype(str).str.strip(),
            'group_name': df_group[3].astype(str).str.strip()
        }).drop_duplicates(subset=['group_id'])
        df_group_clean.to_sql('icd10_groups', conn, if_exists='append', index=False)
        print(f"Berhasil mengimpor {len(df_group_clean)} groups/blocks ICD-10")

    # 3. Load Codes
    file_codes = ICD10_DIR / "icd102019syst_codes.txt"
    if not file_codes.exists():
        print(f"File codes tidak ditemukan: {file_codes}")
        return
        
    try:
        # Load dengan pandas
        df = pd.read_csv(file_codes, sep=';', header=None, encoding='utf-8', 
                         on_bad_lines='skip', low_memory=False)
        
        df_clean = pd.DataFrame({
            'code': df[6].astype(str).str.strip(),
            'description': df[9].astype(str).str.strip(),
            'group_id': df[4].astype(str).str.strip(),
            'chapter_id': df[3].astype(str).str.strip()
        })
        
        # Hapus duplikat kode jika ada
        df_clean = df_clean.drop_duplicates(subset=['code'])
        
        df_clean.to_sql('icd10_codes', conn, if_exists='append', index=False, 
                        chunksize=1000, method='multi')
        print(f"Berhasil mengimpor {len(df_clean)} kode ICD-10")
        
    except Exception as e:
        print(f"Gagal memproses ICD-10 codes: {e}")

def main():
    print(f"Menyiapkan database di: {DB_PATH}")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Hubungkan ke database (akan membuat file baru jika belum ada)
    conn = sqlite3.connect(DB_PATH)
    
    try:
        # Mengaktifkan foreign key constraints untuk sqlite
        conn.execute("PRAGMA foreign_keys = ON")
        
        # Reset database tables agar bisa dijalankan ulang tanpa duplikat
        # Menghapus tabel jika sudah ada sebelum menginisialisasi ulang
        conn.execute("DROP TABLE IF EXISTS drug_interactions")
        conn.execute("DROP TABLE IF EXISTS icd10_codes")
        conn.execute("DROP TABLE IF EXISTS icd10_groups")
        conn.execute("DROP TABLE IF EXISTS icd10_chapters")
        
        init_db(conn)
        
        load_ddinter(conn)
        load_icd10(conn)
        
        print("\nProses ETL selesai!")
        
    finally:
        conn.close()

if __name__ == "__main__":
    main()
