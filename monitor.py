import simpy

class monitor(object):
    def __init__(self, checkfreq, itercount):
        self.check = checkfreq
        self.iter = itercount
        self.blockfrq = 0
        self.operfrq = 0
        self.checkfrq = 0
        self.entryq_sum = 0  # For average queue length
        self.bf_dump = []
        self.op_dump = []
        self.q_dump = []  # List of queue lengths for serial correlation

    def reset(self):
        self.blockfrq = 0
        self.checkfrq = 0
        self.operfrq = 0
        self.entryq_sum = 0
        self.q_dump = []

    def report(self, it):
        self.bf_dump.append(self.blockfrq)
        self.op_dump.append(self.operfrq)

    def run(self, machine, env):
        while True:
            yield env.timeout(self.check)
            self.checkfrq += 1
            entryq_len = len(machine.Preparation.queue)
            self.q_dump.append(entryq_len)  # Collect for ACF
            self.entryq_sum += entryq_len  # For average
            if machine.isblocking:
                self.blockfrq += 1
            if machine.isoperational:
                self.operfrq += 1

    def dump(self):
        print("Blocking frequencies:", self.bf_dump)
        print("Operation frequencies:", self.op_dump)

    def get_avg_entryq(self):
        return self.entryq_sum / self.checkfrq if self.checkfrq > 0 else 0