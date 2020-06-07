from ambulance import *
from patient import *
from charging_station import *
import graph_utils
import copy

class Hospital:
    def __init__(self, pos):
        self.pos = pos
        self.hospital_charging_station = (ChargingStation(self.pos, 2000))

    def get_hospital_pos(self):
        return self.pos

    def get_charging_station(self):
        return self.hospital_charging_station

    #---------------------------------------------------------------------------------
    #           Choose Ambulance to Assign to Pacient Through Heuristics
    #---------------------------------------------------------------------------------

    def choose_ambulance(self, heuristic, ambulances, patient, graph):
        if heuristic == "NCA":
            # (Nearest, Charged and Available)
            # The chosen ambulance is available, has enough energy to deliver the
            # patient to the hospital and is the nearest ambulance to the patient

            patient_pos = patient.get_patient_pos()
            patient_to_hospital = graph_utils.shortest_path(graph, patient_pos, self.pos)[0] # path weight from patient to hospital

            #-- choose an ambulance to pick up patient
            picked_a = None
            picked_ambulance_to_patient = 1000**8 # path weight from ambulance to patient
            for a in ambulances:
                a_to_patient = graph_utils.shortest_path(graph, a.get_pos(), patient_pos)[0] # path weight from ambulance[a] to patient
                            
                if(a.is_available() and a.get_energy_available() >= (a_to_patient + patient_to_hospital) and a_to_patient < picked_ambulance_to_patient):
                    picked_a = a
                    picked_ambulance_to_patient = a_to_patient
                
            #-- assigns ambulance to patient and sets ambulance goal
            if(picked_a):
                patient.set_assigned(True)
                picked_a.set_goal_pacient(patient, picked_ambulance_to_patient, patient_to_hospital)

    
    #---------------------------------------------------------------------------------
    #                                  Execute
    #---------------------------------------------------------------------------------

    def execute(self, ambulances, patients, charging_stations, graph, heuristic):
        #assign ambulance for each patient 
        for p in patients:
            if (not p.is_assigned()): #patient is not assigned, we need to assign him an ambulance
                if heuristic == "NCA":
                    self.choose_ambulance("NCA", ambulances, p, graph) #chooses ambulance and assigns it to the pacient

        #go through ambulances and see if they need to charge (energy below 50% capacity)
        for a in ambulances:
            if a.available and a.energy_available < (a.batery_capacity * 0.5) and a.get_goal() != "get_energy" and a.get_goal() != "go_to_energy":
                (best_distance, best_cs) = graph_utils.closest_charging_station(graph, a.pos, charging_stations)
                a.set_goal_charge(best_distance, best_cs)

            
