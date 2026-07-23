import os
import re
import zipfile
import shutil
import argparse
import requests
import subprocess
from ai_engine.data_pairing.ingest import run_ingest

JOBS_DONE_PATH = "data/jobs_done.txt"
RAW_INCOMING_DIR = "data/raw_incoming"

def check_disk_space(min_gb=40):
    """Kiểm tra dung lượng đĩa trống, trả về True nếu trống >= min_gb."""
    total, used, free = shutil.disk_usage(".")
    free_gb = free / (1024 ** 3)
    print(f"[*] Kiểm tra ổ đĩa: Trống {free_gb:.2f} GB (Ngưỡng yêu cầu: {min_gb} GB)")
    return free_gb >= min_gb

def is_job_done(job_name, before_url, after_url):
    """Kiểm tra xem job hoặc link đã được xử lý chưa bằng cách so khớp chính xác từng dòng."""
    if not os.path.exists(JOBS_DONE_PATH):
        return False
    try:
        with open(JOBS_DONE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                # Mỗi dòng lưu dạng: Job: <job_name> | Before: <before_url> | After: <after_url>
                match = re.match(r"^Job:\s*(.*?)\s*\|\s*Before:\s*(.*?)\s*\|\s*After:\s*(.*?)\s*$", line.strip())
                if match:
                    j_name, b_url, a_url = match.group(1), match.group(2), match.group(3)
                    if job_name == j_name or before_url == b_url or after_url == a_url:
                        return True
    except Exception as e:
        print(f"[!] Cảnh báo lỗi đọc file log jobs_done.txt: {str(e)}")
    return False

def mark_job_done(job_name, before_url, after_url):
    """Ghi nhận job đã xử lý thành công."""
    os.makedirs(os.path.dirname(JOBS_DONE_PATH), exist_ok=True)
    with open(JOBS_DONE_PATH, "a", encoding="utf-8") as f:
        f.write(f"Job: {job_name} | Before: {before_url} | After: {after_url}\n")

def download_gdrive_folder(url, target_dir):
    """Tải thư mục Google Drive sử dụng gdown CLI có cơ chế tự động Retry."""
    import time
    os.makedirs(target_dir, exist_ok=True)
    
    cmd = ["gdown", "--folder", url, "-O", target_dir]
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        print(f"[*] Bắt đầu tải Google Drive Folder (Lần thử {attempt}/{max_retries}) về: {target_dir}")
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("[✓] gdown download hoàn tất.")
            return True
        except subprocess.CalledProcessError as e:
            print(f"[!] Cảnh báo: Tải Google Drive thất bại ở lần thử {attempt}: {e}")
            if attempt < max_retries:
                print("[*] Chờ 5 giây trước khi tải lại...")
                time.sleep(5)
    print("[✗] Đã thử tải Google Drive 3 lần nhưng đều thất bại.")
    return False

def download_dropbox_folder(url, target_dir):
    """Tải thư mục Dropbox bằng cách chuyển link sang direct download zip."""
    print(f"[*] Bắt đầu tải Dropbox Folder về: {target_dir}")
    os.makedirs(target_dir, exist_ok=True)
    
    # Đổi www.dropbox.com sang dl.dropboxusercontent.com để tránh HTML redirect
    download_url = url
    if "dl=0" in download_url:
        download_url = download_url.replace("dl=0", "dl=1")
    elif "sp=0" in download_url:
        download_url = download_url.replace("sp=0", "dl=1")
    elif "dl=1" not in download_url:
        if "?" in download_url:
            download_url = download_url + "&dl=1"
        else:
            download_url = download_url + "?dl=1"
            
    temp_zip_path = os.path.join(target_dir, "temp_dropbox.zip")
    
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        r = requests.get(download_url, stream=True, headers=headers, timeout=60, allow_redirects=True)
        r.raise_for_status()
        
        # Kiểm tra content-type
        content_type = r.headers.get("content-type", "")
        if "text/html" in content_type:
            print("[✗] Lỗi: Link Dropbox trả về trang HTML thay vì file download trực tiếp. Có thể link đã hết hạn hoặc ở chế độ riêng tư.")
            return False
            
        with open(temp_zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    
        print("[✓] Tải file zip của Dropbox thành công. Đang giải nén...")
        
        # Giải nén
        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
            
        # Đợi ngắn và retry để file handle được giải phóng hoàn toàn trên Windows
        import time
        time.sleep(0.5)
        for attempt in range(5):
            try:
                if os.path.exists(temp_zip_path):
                    os.remove(temp_zip_path)
                break
            except PermissionError:
                time.sleep(0.5)
                
        print("[✓] Giải nén Dropbox hoàn tất.")
        return True
    except Exception as e:
        print(f"[✗] Lỗi khi tải/giải nén Dropbox: {str(e)}")
        # Đợi ngắn và retry để file handle được giải phóng hoàn toàn khi có lỗi
        import time
        time.sleep(0.5)
        for attempt in range(5):
            try:
                if os.path.exists(temp_zip_path):
                    os.remove(temp_zip_path)
                break
            except PermissionError:
                time.sleep(0.5)
        return False

def detect_before_after_dirs(temp_a, temp_b):
    """
    Tự động phân loại Before/After dựa trên nội dung:
    Thư mục nào chứa file RAW (.CR3/.DNG/.ARW...) là Before.
    Thư mục còn lại chứa JPG/PNG là After.
    Nếu cấu trúc mơ hồ (cả 2 đều có RAW hoặc cả 2 đều 0 RAW) -> BÁO LỖI DỪNG.
    """
    raw_extensions = {".cr3", ".dng", ".arw", ".nef", ".orf", ".rw2"}
    
    def count_raw_files(directory):
        count = 0
        if not os.path.exists(directory):
            return count
        for root, _, files in os.walk(directory):
            for f in files:
                if any(f.lower().endswith(ext) for ext in raw_extensions):
                    count += 1
        return count

    raws_in_a = count_raw_files(temp_a)
    raws_in_b = count_raw_files(temp_b)
    
    print(f"[*] Quét nội dung thư mục tạm:")
    print(f"    - Thư mục Temp A: có {raws_in_a} file RAW")
    print(f"    - Thư mục Temp B: có {raws_in_b} file RAW")
    
    if raws_in_a > 0 and raws_in_b == 0:
        return temp_a, temp_b
    elif raws_in_b > 0 and raws_in_a == 0:
        return temp_b, temp_a
    else:
        raise ValueError(
            f"[✗] LỖI CẤU TRÚC MƠ HỒ: Không thể tự động phân định Before/After. "
            f"Temp A có {raws_in_a} RAW, Temp B có {raws_in_b} RAW. Dừng tác vụ."
        )

def fetch_and_process_job(job_name, before_url, after_url, keep_raw=False, force=False):
    """
    Tải job từ link Cloud, tự động nhận diện before/after, chạy Ingest, và dọn dẹp RAW.
    """
    # 1. Kiểm tra đĩa trống
    if not check_disk_space(40):
        print("[✗] DỪNG TÁC VỤ: Ổ đĩa VPS còn dưới 40GB trống. Không thể tải thêm dữ liệu để bảo vệ hệ thống!")
        return False
        
    # 2. Kiểm tra chống trùng
    if not force and is_job_done(job_name, before_url, after_url):
        print(f"[*] BỎ QUA: Job '{job_name}' hoặc link này đã được xử lý trước đó. Dùng --force để chạy lại.")
        return True
        
    job_dir = os.path.join(RAW_INCOMING_DIR, job_name)
    temp_a = os.path.join(job_dir, "temp_a")
    temp_b = os.path.join(job_dir, "temp_b")
    
    # Xóa sạch thư mục job nếu có dở
    if os.path.exists(job_dir):
        shutil.rmtree(job_dir, ignore_errors=True)
    os.makedirs(job_dir, exist_ok=True)
    
    # 3. Tải Link 1 và Link 2 về thư mục tạm temp_a và temp_b
    # Tải Link 1 (trước) -> temp_a
    link1_success = False
    if "drive.google.com" in before_url:
        link1_success = download_gdrive_folder(before_url, temp_a)
    elif "dropbox.com" in before_url:
        link1_success = download_dropbox_folder(before_url, temp_a)
    else:
        print(f"[✗] Link 1 không được hỗ trợ hoặc không nhận diện được nhà cung cấp: {before_url}")
        
    # Tải Link 2 (sau) -> temp_b
    link2_success = False
    if "drive.google.com" in after_url:
        link2_success = download_gdrive_folder(after_url, temp_b)
    elif "dropbox.com" in after_url:
        link2_success = download_dropbox_folder(after_url, temp_b)
    else:
        print(f"[✗] Link 2 không được hỗ trợ hoặc không nhận diện được nhà cung cấp: {after_url}")
        
    if not link1_success or not link2_success:
        print("[✗] Tải dữ liệu từ Cloud thất bại. Để nguyên thư mục dở để debug, KHÔNG dọn dẹp.")
        return False
        
    # 4. Tự nhận diện Before/After từ temp_a và temp_b
    try:
        before_src, after_src = detect_before_after_dirs(temp_a, temp_b)
    except ValueError as e:
        print(str(e))
        return False
    
    before_dest = os.path.join(job_dir, "before")
    after_dest = os.path.join(job_dir, "after")
    
    # Move sang đúng cấu trúc before/after
    shutil.move(before_src, before_dest)
    shutil.move(after_src, after_dest)
    
    # Xóa thư mục temp rỗng còn lại
    if os.path.exists(temp_a):
        shutil.rmtree(temp_a, ignore_errors=True)
    if os.path.exists(temp_b):
        shutil.rmtree(temp_b, ignore_errors=True)
        
    print("[✓] Nhận diện và sắp xếp Before/After hoàn tất.")
    
    # 5. Gọi pipeline Ingest SỬA 3
    print(f"[*] Đang thực thi Ingest cho Job: {job_name}...")
    try:
        run_ingest(reset=False, before_root=before_dest, after_root=after_dest, job_name=job_name)
    except Exception as e:
        print(f"[✗] Lỗi xảy ra khi chạy Ingest: {str(e)}. Giữ nguyên RAW để debug.")
        return False
        
    # 6. Purge RAW (Mặc định xóa raw incoming để giải phóng ổ)
    if not keep_raw:
        print(f"[*] Đang xóa thư mục RAW incoming: {job_dir} để giải phóng ổ đĩa...")
        try:
            shutil.rmtree(job_dir, ignore_errors=True)
            # Thêm bước kiểm tra xóa sạch
            if not os.path.exists(job_dir):
                print("[✓] Xóa RAW incoming hoàn tất sạch sẽ.")
            else:
                print("[!] Cảnh báo: Thư mục raw_incoming vẫn tồn tại sau khi rmtree.")
        except Exception as e:
            print(f"[!] Cảnh báo: Không thể xóa thư mục RAW tạm: {str(e)}")
    else:
        print("[*] Tùy chọn --keep-raw được bật. Giữ lại thư mục RAW incoming.")
        
    # 7. Ghi nhận log đã hoàn thành
    mark_job_done(job_name, before_url, after_url)
    print(f"[✓] JOB {job_name} HOÀN THÀNH XUẤT SẮC!")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoHDR Job Fetcher - Download cloud folders, pair and ingest")
    parser.add_argument("--name", type=str, required=True, help="Tên của job (Property/Căn hộ)")
    parser.add_argument("--before", type=str, required=True, help="Link chứa ảnh RAW/before (Dropbox hoặc Google Drive)")
    parser.add_argument("--after", type=str, required=True, help="Link chứa ảnh JPG/after (Dropbox hoặc Google Drive)")
    parser.add_argument("--keep-raw", action="store_true", help="Giữ lại thư mục RAW incoming tải về sau khi ingest")
    parser.add_argument("--force", action="store_true", help="Ép buộc chạy lại kể cả khi job đã có trong logs")
    args = parser.parse_args()
    
    fetch_and_process_job(
        job_name=args.name,
        before_url=args.before,
        after_url=args.after,
        keep_raw=args.keep_raw,
        force=args.force
    )
