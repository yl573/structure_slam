import time

class RunningAverageTimer:

    def __init__(self):
        self.times = []
        self.start_time = None

    def start(self):
        assert self.start_time is None
        self.start_time = time.time()

    def stop(self):
        assert self.start_time is not None
        t = time.time() - self.start_time
        self.times.append(t)
        self.start_time = None

    def running_avg(self, n=20):
        times = self.times[-n:]
        return sum(times) / len(times)

    def __enter__(self):
        self.start()

    def __exit__(self, type, value, traceback):
        self.stop()
        print(f'avg time: {self.running_avg()}')

