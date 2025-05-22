import os
import hashlib
from PIL import Image
from icrawler.builtin import GoogleImageCrawler

# prompt with profession keywords
# The keywords are used to search for images of different professions
profession_keywords = {
    "doctor": [
        "doctor", "female doctor", "male doctor", "doctor in hospital", "doctor portrait",
        "professional doctor", "doctor with stethoscope", "young doctor", "doctor photo", "male doctor portrait"
    ],
    "nurse": [
        "nurse", "female nurse", "male nurse", "nurse working", "nurse in clinic",
        "nurse with patient", "professional nurse", "nurse portrait", "young nurse", "nurse photo"
    ],
    "teacher": [
        "teacher", "female teacher", "male teacher", "teacher in classroom", "teacher portrait",
        "school teacher", "high school teacher", "teacher with students", "teacher at blackboard",
        "teaching photo", "teacher at whiteboard", "elementary school teacher", "professional teacher"
    ],
    "lawyer": [
        "lawyer", "female lawyer", "male lawyer", "lawyer in office", "lawyer at desk",
        "lawyer portrait", "professional lawyer", "lawyer working", "lawyer with files", "young lawyer"
    ],
    "engineer": [
        "engineer", "female engineer", "male engineer", "engineer working", "engineer with helmet",
        "software engineer", "mechanical engineer", "civil engineer", "engineer at desk", "young engineer"
    ],
    "hairdresser": [
        "hairdresser", "female hairdresser", "male hairdresser", "hairdresser cutting hair", "hair stylist",
        "barber", "professional hairdresser", "hair salon worker", "hairdresser with scissors", "hairdresser portrait"
    ],
    "baker": [
        "baker", "female baker", "male baker", "baker baking", "baker in kitchen",
        "pastry chef", "bread maker", "baker with apron", "baker holding bread", "bakery worker"
    ],
    "waiter": [
        "waiter", "waitress", "male waiter", "female waiter", "waiter with tray",
        "restaurant staff", "cafe worker", "waiter serving food", "young waiter", "waiter portrait"
    ],
    "veterinarian": [
        "veterinarian", "female veterinarian", "male veterinarian", "vet with animal", "animal doctor",
        "vet clinic worker", "veterinarian examining pet", "professional vet", "veterinarian in clinic", "vet portrait"
    ],
    "painter": [
        "painter", "female painter", "male painter", "house painter", "painter with brush",
        "painter in uniform", "wall painter", "painter working", "professional painter", "painter portrait"
    ],
}

MAX_NUM = 2000
SAVE_ROOT = './Crawler_data'
MIN_RESOLUTION = (128, 128)

def remove_undesired_images(folder):
    seen = set()
    count = 1
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        try:
            with open(fpath, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            if file_hash in seen:
                os.remove(fpath)
                continue
            seen.add(file_hash)

            with Image.open(fpath) as img:
                img.verify()

            with Image.open(fpath) as img:
                w, h = img.size
            if w < MIN_RESOLUTION[0] or h < MIN_RESOLUTION[1]:
                os.remove(fpath)
                continue

            new_name = f"{count:08d}.jpg"
            os.rename(fpath, os.path.join(folder, new_name))
            count += 1

        except Exception:
            os.remove(fpath)

    print(f"[INFO] {os.path.basename(folder)}: {count - 1} images saved.\n")

def download_for_profession(profession, keywords):
    folder = os.path.join(SAVE_ROOT, profession.replace(" ", "_"))
    os.makedirs(folder, exist_ok=True)

    for keyword in keywords:
        print(f"   -> Searching: {keyword}")
        crawler = GoogleImageCrawler(storage={"root_dir": folder})
        crawler.crawl(
            keyword=keyword,
            max_num=MAX_NUM,
            filters={"type": "photo"},
            # filters={},
            file_idx_offset=0
        )

    remove_undesired_images(folder)

if __name__ == "__main__":
    for profession, keywords in profession_keywords.items():
        print(f"\n==> Downloading: {profession}")
        download_for_profession(profession, keywords)

    print("\nAll downloads complete.")
