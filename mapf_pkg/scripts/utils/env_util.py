from functools import wraps

class EventCounter(object):
    def __init__(self, schedules=None, matters=None):
        self.counter = 0
        self.schedules = schedules
        self.matters = matters
        if not isinstance(schedules, list) or not isinstance(matters, list):
            raise ValueError("schedules and matter must be list.")
        if not all([schedules, matters]):
            raise ValueError("Need to set schedules and matters both.")

    def __call__(self, f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if self.schedules:
                if self.counter == self.schedules[0]:
                    self.schedules.pop(0)
                    if self.matters:
                        self.matters.pop(0)
            wrapper.counter = self.counter
            wrapper.schedule = self.schedules[0] if self.schedules else None
            wrapper.matter = self.matters[0] if self.matters else None
            self.counter += 1
            return f(*args, **kwargs)
        return wrapper

if __name__ == '__main__':
    @EventCounter([3, 6, 9, 12], [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)])
    def reset():
        print(reset.matter)

    for _ in range(20):
        reset()
