# python_sysinfo

## platform_info
### check_os
```python
import platform
print(platform.system(), platform.release(), platform.machine())
```

### check_python
```python
import sys
print('Python', sys.version)
print('Executable:', sys.executable)
```

## resource_info
### check_cpu
```python
import os
print('CPUs:', os.cpu_count())
```

### check_memory
```python
mem = {}
for line in open('/proc/meminfo'):
    k, v = line.split(':')
    mem[k] = v.strip()
total = int(mem['MemTotal'].split()[0]) // 1024
avail = int(mem['MemAvailable'].split()[0]) // 1024
print(f'RAM: {total} MB total, {avail} MB available')
```

### check_load
```python
import os
la = os.getloadavg()
print(f'Load average: {la[0]:.2f}  {la[1]:.2f}  {la[2]:.2f}')
```
