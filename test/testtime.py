from datetime import datetime

now = datetime.now()

format_time = now.strftime(f'%Y-%m-%d_%H-%M-{int(now.second)}')

print(format_time)