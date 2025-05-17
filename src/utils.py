def time_to_seconds(time_str: str) -> int:
    h, m, s = map(int, time_str.split(":"))
    return int(h * 3600 + m * 60 + s)


def seconds_to_time(seconds: int) -> list:
    h = seconds//3600
    seconds = seconds%3600
    m = seconds//60
    s = seconds%60
    return [h, m, s]