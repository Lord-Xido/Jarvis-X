class Debugger:
    def __init__(self, vm):
        self.vm = vm
        self.breakpoints = set()

    def set_breakpoint(self, ip):
        self.breakpoints.add(ip)

    def check(self):
        return self.vm.regs["IP"] in self.breakpoints
