class ChargingStation:
    def __init__(self, pos, capacity = 5):
        self.pos = pos
        self.capacity = capacity
        self.nr_charging_ambulances = 0

    def get_charging_station_pos(self):
        return self.pos

    def get_capacity(self):
        return self.capacity

    def get_pos(self):
        return self.pos

    def has_enough_capacity(self):
        if self.nr_charging_ambulances < self.capacity:
            return True
        else:
            return False

    def stop_charge(self):
        self.nr_charging_ambulances -= 1

    def start_charge(self):
        self.nr_charging_ambulances += 1

    def recharge_energy(self, id, ambulance):
        available = False
        ambulance.energy_available += 5 # Each TIK recharges 5 units of energy
        if (ambulance.energy_available >= ambulance.batery_capacity): #To avoid having energy > capacity
            energy_available = ambulance.batery_capacity
            print("Ambulance #" + str() + " is now fully charged" + ". Battery %" + str(ambulance.get_energy_available()))
            available = True
            
        elif (ambulance.energy_available >= ambulance.batery_capacity * 0.7): #battery at 70% switches availability
            available = True
        return available 