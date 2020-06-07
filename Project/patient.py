class Patient:
    def __init__(self, id, pos, emergency_priority, emergency_time, time_of_occurrence):
        self.id = id
        self.pos = pos
        self.emergency_priority = emergency_priority # 1 low , 2 medium, 3 high
        self.assigned = False
        self.emergency_time = emergency_time
        self.time_of_occurrence = time_of_occurrence

    def get_patient_pos(self):
        return self.pos

    def get_id(self):
        return self.id

    def set_assigned(self, is_assigned):
        self.assigned = is_assigned

    def is_assigned(self):
        return self.assigned

    def get_emergency_time(self):
        return self.emergency_time

    def get_time_of_occurrence(self):
        return self.time_of_occurrence

    def get_emergency_priority(self):
        return self.emergency_priority