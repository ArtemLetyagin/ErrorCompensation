def format_bytes(n):
    if n == 0:
        return "0B"
    units = ["B", "KB", "MB", "GB"]
    x = float(n)
    i = 0
    while x >= 1024 and i < len(units) - 1:
        x /= 1024
        i += 1
    return f"{x:.2f} {units[i]}" if i > 0 else f"{int(x)}B"
