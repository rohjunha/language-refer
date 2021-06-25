import hashlib


def generate_hash(
        scan_id: str,
        instance_id: int) -> int:
    return int(hashlib.sha256((str(scan_id) + str(instance_id)).encode('utf-8')).hexdigest(), 16) % (10 ** 16)
