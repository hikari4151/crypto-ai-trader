"""API Key 加解密：Fernet 对称加密，主密钥来自 MASTER_KEY 或自动生成的 data/master.key。"""
import base64
import hashlib
import os
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from config.settings import DATA_DIR

KEY_FILE = DATA_DIR / "master.key"


def _master_key() -> bytes:
    master = os.environ.get("MASTER_KEY", "")
    if not master:
        # 优先读本地主密钥文件
        if KEY_FILE.exists():
            master = KEY_FILE.read_text().strip()
        else:
            master = Fernet.generate_key().decode()
            KEY_FILE.write_text(master)
            try:
                os.chmod(KEY_FILE, 0o600)  # 仅当前用户可读写
            except OSError:
                pass
    # 归一化为合法 Fernet key（32 字节 urlsafe-base64）
    digest = hashlib.sha256(master.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


_fernet = None


def _get_fernet() -> Fernet:
    global _fernet
    if _fernet is None:
        _fernet = Fernet(_master_key())
    return _fernet


def encrypt(plaintext: str) -> str:
    if not plaintext:
        return ""
    return _get_fernet().encrypt(plaintext.encode("utf-8")).decode()


def decrypt(token: str) -> str:
    if not token:
        return ""
    try:
        return _get_fernet().decrypt(token.encode()).decode()
    except InvalidToken:
        raise ValueError("解密失败：主密钥不匹配或数据已损坏")


def mask(plaintext: str) -> str:
    """返回掩码后的密钥用于前端展示，绝不明文返回。"""
    if not plaintext or len(plaintext) < 8:
        return ""
    return plaintext[:4] + "****" + plaintext[-4:]