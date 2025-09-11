from pathlib import Path
import json

# отдельное хранилище для voice_car
PROFILES_FILE = Path("games/voice_car/components/profiles_voicecar.json")

def _ensure_dir():
    PROFILES_FILE.parent.mkdir(parents=True, exist_ok=True)

def _default_audio():
    return {"mic_device_index": None, "silence_threshold_db": 50.0}

def _default_settings():
    return {
        "pitch_ranges": {
            "low": [150, 199],
            "normal": [200, 250],
            "high": [251, 300],
        },
        "base_duration": 4.0,
        "max_out_of_zone": 0.5,
        "blocks_to_silent": 2,
    }

def load_profiles():
    if PROFILES_FILE.exists() and PROFILES_FILE.read_text(encoding="utf-8").strip():
        data = json.loads(PROFILES_FILE.read_text(encoding="utf-8"))
        for p in data.get("profiles", []):
            p.setdefault("audio", _default_audio())
            p.setdefault("settings", _default_settings())
            p.setdefault("tasks", [])
        return data
    return {"profiles": []}

def save_profiles(data: dict):
    _ensure_dir()
    PROFILES_FILE.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")

def find_profile(data: dict, name: str):
    for p in data.get("profiles", []):
        if p.get("name") == name:
            return p
    return None

# -------- профили --------
def add_profile(name: str):
    if not name or not name.strip():
        raise ValueError("Имя профиля пустое")
    data = load_profiles()
    if find_profile(data, name):
        raise ValueError("Профиль с таким именем уже существует")
    data.setdefault("profiles", []).append({
        "name": name,
        "audio": _default_audio(),
        "settings": _default_settings(),
        "tasks": []
    })
    save_profiles(data)

def delete_profile(name: str):
    data = load_profiles()
    data["profiles"] = [p for p in data.get("profiles", []) if p.get("name") != name]
    save_profiles(data)

def rename_profile(old_name: str, new_name: str):
    if not new_name or not new_name.strip():
        raise ValueError("Новое имя пустое")
    data = load_profiles()
    if old_name == new_name:
        return
    if find_profile(data, new_name):
        raise ValueError("Профиль с таким именем уже существует")
    p = find_profile(data, old_name)
    if not p:
        raise ValueError("Профиль не найден")
    p["name"] = new_name
    save_profiles(data)

# -------- задачи --------
def get_tasks(profile_name: str):
    p = find_profile(load_profiles(), profile_name)
    return p.get("tasks", []) if p else []

def set_tasks(profile_name: str, tasks: list):
    data = load_profiles()
    p = find_profile(data, profile_name)
    if not p:
        raise ValueError("Профиль не найден")
    p["tasks"] = tasks
    save_profiles(data)

def upsert_task(profile_name: str, new_task: dict):
    data = load_profiles()
    p = find_profile(data, profile_name)
    if not p:
        raise ValueError("Профиль не найден")
    tasks = p.setdefault("tasks", [])
    for i, t in enumerate(tasks):
        if t.get("name") == new_task.get("name"):
            tasks[i] = new_task
            break
    else:
        tasks.append(new_task)
    save_profiles(data)

# -------- аудио --------
def get_audio(profile_name: str):
    p = find_profile(load_profiles(), profile_name)
    return p.get("audio", _default_audio()) if p else _default_audio()

def set_audio(profile_name: str, *, mic_device_index=None, silence_threshold_db=None):
    data = load_profiles()
    p = find_profile(data, profile_name)
    if not p:
        raise ValueError("Профиль не найден")
    audio = p.setdefault("audio", _default_audio())
    if mic_device_index is not None:
        audio["mic_device_index"] = mic_device_index
    if silence_threshold_db is not None:
        audio["silence_threshold_db"] = float(silence_threshold_db)
    save_profiles(data)

# -------- общие настройки профиля (диапазоны и т.п.) --------
def get_settings(profile_name: str):
    p = find_profile(load_profiles(), profile_name)
    return p.get("settings", _default_settings()) if p else _default_settings()

def set_settings(profile_name: str, settings: dict):
    data = load_profiles()
    p = find_profile(data, profile_name)
    if not p:
        raise ValueError("Профиль не найден")
    p["settings"] = settings
    save_profiles(data)
