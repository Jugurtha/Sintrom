class SintromDosage:
    dosage = {
        'Sunday': 0,
        'Monday': 0,
        'Tuesday': 0,
        'Wednesday': 0,
        'Thursday': 0,
        'Friday': 0,
        'Saturday': 0
    }

    def __init__(self, dosage: tuple):
        self.dosage['Sunday'] = dosage[0]
        self.dosage['Monday'] = dosage[1]
        self.dosage['Tuesday'] = dosage[2]
        self.dosage['Wednesday'] = dosage[3]
        self.dosage['Thursday'] = dosage[4]
        self.dosage['Friday'] = dosage[5]
        self.dosage['Saturday'] = dosage[6]

    def dosage_to_tuple(self) -> tuple:
        return tuple(self.dosage.values())
