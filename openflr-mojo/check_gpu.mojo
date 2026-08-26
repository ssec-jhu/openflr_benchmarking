from std.sys import has_accelerator
from max.gpu.host import DeviceContext

def main() raises:
    comptime if not has_accelerator():
        print("No GPU found")
    else:
        var ctx = DeviceContext()
        print("device name:", ctx.name())
