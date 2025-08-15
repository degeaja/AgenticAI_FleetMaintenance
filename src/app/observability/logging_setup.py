# logging_setup.py
import logging, sys, json, os
from logging.handlers import RotatingFileHandler
from datetime import datetime

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {
            "ts": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # attach extras (avoid private attrs)
        for k, v in getattr(record, "__dict__", {}).items():
            if k.startswith(("_", "args", "msg", "level")): 
                continue
            if k in base: 
                continue
            try:
                json.dumps({k: v})
                base[k] = v
            except Exception:
                base[k] = str(v)
        return json.dumps(base, ensure_ascii=False)

def build_logger(
    name: str = "fleet",
    level: str = os.getenv("LOG_LEVEL", "INFO"),
    to_file: bool = True,
    file_path: str = os.getenv("LOG_FILE", "logs/fleet_agent.log"),
    max_bytes: int = 5_000_000,
    backup_count: int = 3,
) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:  # prevent double handlers in reload
        return logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    fmt = JsonFormatter()
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if to_file:
        fh = RotatingFileHandler(file_path, maxBytes=max_bytes, backupCount=backup_count)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    # quiet noisy libs (optional)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("watchfiles").setLevel(logging.WARNING)
    return logger

logger = build_logger()
