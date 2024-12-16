class Job:
    def __init__(self, firm, wage):
        self.firm = firm
        self.wage = wage

    def accept(self):
        self.firm.num_workers += 1
        self.firm.m += 1

    def __lt__(self, other):
        return self.wage < other.wage