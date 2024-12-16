from enum import Enum

class WorkerState(Enum):
    UNEMPLOYED = 0
    EMPLOYED = 1

# class Worker:
#     def __init__(self, state = WorkerState.UNEMPLOYED):
#         self.state = state

#     def employ(self, job):
#         job.accept()
#         self.state = WorkerState.EMPLOYED

#     def dismiss(self):
#         self.state = WorkerState.UNEMPLOYED
