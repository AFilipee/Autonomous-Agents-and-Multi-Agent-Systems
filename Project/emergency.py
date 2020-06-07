import networkx as nx
import sys
from os import system
from random import *
import matplotlib.pylab as plt

import graph_utils
from ambulance import *
from patient import *
from hospital import *
from charging_station import *


#---------------------------------------------------------------------------------
#---------------------------------- EMERGENCY ------------------------------------
#---------------------------------------------------------------------------------
class Emergency(object):

    def __init__(self, n_nodes=15, p=0.2, min_weight=1, max_weight=5, nr_ambulances=2, nr_charging_stations=2):
        #tik
        self.tik = 0

        #user input variables
        self.nr_ambulances = nr_ambulances
        self.nr_charging_stations = nr_charging_stations
        
        #graph
        self.n_nodes = n_nodes       #number of nodes
        self.p = p                   #probability for edge (between nodes) creation
        self.min_weight = min_weight #min weight (= time) between two nodes
        self.max_weight = max_weight #max weight (= time) between two nodes
        self.g = None

        #agents
        self.ambulances = []
        self.hospital = None
        self.charging_stations = []
        self.patients = []
        self.patient_generation_list = []
        self.training_patient_generation_list = []

        #counters
        self.patient_counter = 0

        #simulations
        self.success = 0 #counter of success
        self.failed  = 0 #counter of failure

        #metrics
        self.priority      = [0,0,0] #low, moderate, high
        self.priority_time = [0,0,0]

        #reinforcement learning
        self.action_selection = Ambulance.ActionSelection.selective
        self.learning_approach = Ambulance.LearningApproach.QLearning
        self.n_training_patients = self.n_nodes * 10

    #---------------------------------------------------------------------------------
    #                                   Setters 
    #---------------------------------------------------------------------------------

    def set_n_nodes(self, n):
        self.n_nodes = n
        if (n >= 15):
            self.nr_ambulances = randint(round(n/10), round(n/5))
    
    def set_min_weight(self, min_w):
        self.min_weight = min_w

    def set_max_weight(self, max_w):
        self.max_weight = max_w

    def set_nr_ambulances(self, nr_ambulances):
        self.nr_ambulances = nr_ambulances
        
    def set_nr_charging_stations(self, nr_charging_stations):
        self.nr_charging_stations = nr_charging_stations

    def set_graph(self, g):
        self.g = g

    def set_priority(self, priority):
        self.priority = priority

    def set_priority_time(self, p_time):
        self.priority_time = p_time

    def set_action_selection(self, a_s):
        if a_s.lower() == "egreedy":    # avoids case sensitive
            self.action_selection = Ambulance.ActionSelection.eGreedy
        elif a_s.lower() == "softmax":
            self.action_selection = Ambulance.ActionSelection.softMax
        elif a_s.lower() == "selective":
            self.action_selection = Ambulance.ActionSelection.selective

    def set_learning_approach(self, l_a):
        if l_a.lower() == "qlearning":
            self.learning_approach = Ambulance.LearningApproach.QLearning
        elif l_a.lower() == "sarsa":
            self.learning_approach = Ambulance.LearningApproach.SARSA

    def set_n_training_patients(self, n):
        self.n_training_patients = n

    #---------------------------------------------------------------------------------
    #                                  TIKs
    #---------------------------------------------------------------------------------

    def get_tik(self):
        return self.tik

    def increment_tik(self):
        self.tik += 1

    #---------------------------------------------------------------------------------
    #                        Build Graph and Populate
    #---------------------------------------------------------------------------------

    def populate_charging_stations(self):
        # sets the charging stations position in the graph 
        l = []
        l.extend(range(0, self.n_nodes))
        l.remove(self.hospital.get_hospital_pos())
        stations_pos = sample(l, self.nr_charging_stations)
        for pos in stations_pos:
            graph_utils.set_charging_station(pos)
            self.charging_stations.append(ChargingStation(pos))

    def populate_patients(self, nr_tiks, number_of_patients):
        #if patient list is empty or number_of_patients has changed --> populate again
        if (not self.patient_generation_list or len(self.patient_generation_list) != number_of_patients):
            self.patient_generation_list = sample(range(nr_tiks), number_of_patients)
            self.patient_generation_list.sort()

    def populate_training_patients(self, n_training_patients):
        self.training_patient_generation_list = sample(range(n_training_patients*10), n_training_patients)
        self.training_patient_generation_list.sort()

    def build_graph(self):
        g = graph_utils.generate_weighted_binomial_graph(self.n_nodes, self.p, self.min_weight, self.max_weight)
   
        # sets the hospital position in the graph
        hospital_location = randint(0, self.n_nodes - 1)
        graph_utils.set_hospital(hospital_location)
        self.hospital = Hospital(hospital_location)

        return g

    #---------------------------------------------------------------------------------
    #                             Agent Generators
    #---------------------------------------------------------------------------------

    def generate_ambulances(self, hospital, nr_ambulances=2, energy=100, capacity = 1, g = None, pos_list = None,
                            a_s=Ambulance.ActionSelection.selective, l_a=Ambulance.LearningApproach.QLearning):
        for i in range(nr_ambulances):
            position = self.hospital.get_hospital_pos()
            self.ambulances.append(Ambulance(len(self.ambulances), position, pos_list, energy, capacity, hospital, g, a_s, l_a))

    def generate_patient(self, average_weight ,time=0, print_info=True):
        nok = True
        while(nok):
            random_pos = randint(0, self.n_nodes-1)
            if(random_pos != self.hospital.get_hospital_pos()):  
                nok = False

        if(self.priority == [0,0,0]):        
            emergency_priority = randint(1,3)
        else:
            population = [1, 2, 3]
            emergency_priority = choices(population, self.priority)[0]

        if(self.priority_time == [0,0,0]):
            time = randint(round(average_weight/emergency_priority)*4 , round(average_weight/emergency_priority)*6) 
        else:
            time = self.priority_time[emergency_priority - 1]

        self.patients.append( Patient(self.patient_counter, random_pos, emergency_priority, time, self.get_tik()) )
        if(print_info):
            print("Generated pacient#"+ str(self.patient_counter) +" with priority #" + str(emergency_priority) + " and emergency time #" + str(time) + " TIKs")
        self.patient_counter += 1

    #---------------------------------------------------------------------------------
    #                                Reset Simulator
    #---------------------------------------------------------------------------------

    def reset_before_simulate(self):
        #agents
        self.ambulances = []
        self.charging_stations = []
        self.patients = []

        #counters
        self.patient_counter = 0
        self.tik = 0
        self.success = 0
        self.failed  = 0

        
    #---------------------------------------------------------------------------------
    #                       Run Simulation #1 -- Hospital Decides
    #---------------------------------------------------------------------------------

    def run_hospital_decide_simulation(self, heuristic, nr_tiks=200):
        nr_tiks_per_pacient = self.max_weight * 3

        self.reset_before_simulate()
        # Step 1: generate the rest of the agents and assign their postitions on the graph
        self.populate_charging_stations()
        self.generate_ambulances(hospital = self.hospital, nr_ambulances = self.nr_ambulances)
        number_of_patients = round(nr_tiks/nr_tiks_per_pacient)
        self.populate_patients(nr_tiks, number_of_patients)

        # Step 2: go through a series of iterations
        while((self.get_tik() < nr_tiks) or self.patients):# or self.patients != []):

            print("\n  TIK #" + str(self.get_tik()))

            if(self.patient_counter < number_of_patients and self.get_tik() == self.patient_generation_list[self.patient_counter]):
                self.generate_patient((self.min_weight + self.max_weight)/2)

            # 1) hospital assigns goals to each ambulance
            self.hospital.execute(self.ambulances, self.patients, self.charging_stations, self.g, heuristic)

            # 2) each ambulance executes3
            for a in self.ambulances:
                o = a.execute(self.hospital.get_hospital_pos())
                if (o == "picked_up_success"): 
                    self.success += 1
                    self.patients.remove(a.get_patient())
                elif (o == "picked_up_failed"):
                    self.failed += 1
                    self.patients.remove(a.get_patient())

            self.increment_tik()
        graph_utils.draw_graph(self.g)

        print("\n\nThe simulation ended with " + str(self.success) + " cases of success and " + str(self.failed) + " cases of failure")
        


    #---------------------------------------------------------------------------------
    #                       Run Simulation #2 -- Ambulances Decide
    #---------------------------------------------------------------------------------

    def run_cooperative_ambulances_simulation(self, nr_tiks=200,a_s=Ambulance.ActionSelection.selective, l_a=Ambulance.LearningApproach.QLearning):
        self.reset_before_simulate()

        nr_tiks_per_pacient = self.max_weight*3

        # Step 1: generate the rest of the agents and assign their postitions on the graph
        self.populate_charging_stations()
        self.generate_ambulances(hospital = self.hospital, nr_ambulances = self.nr_ambulances,
                                 g = self.g, pos_list=self.n_nodes, a_s=Ambulance.ActionSelection.selective,
                                 l_a=Ambulance.LearningApproach.QLearning)

        number_of_patients = round(nr_tiks/nr_tiks_per_pacient)
        self.populate_patients(nr_tiks, number_of_patients)
        # Step 2: let agents learn the positions where patients have more case and higher emergency
        self.populate_training_patients(self.n_training_patients)
        print("\nNumber of patients to be generated: " + str(self.n_training_patients))
        print("\nLearning in progress...")
        print("\nThis might take a while...")
        for i in range(self.n_training_patients * 10):
            if self.patient_counter < self.n_training_patients and i == self.training_patient_generation_list[self.patient_counter]:
                self.generate_patient((self.min_weight + self.max_weight)/2, print_info=False)

            for a in self.ambulances:
                a.getLearningMetrics(self.g, self.patients, self.hospital)
                a.executeLearning()

        self.patients.clear()
        self.patient_counter = 0
        for a in self.ambulances:
            a.set_exploitation()
        print("\nLearning completed\n\n")

        # Step 3: go through a series of iterations
        picked = 0
        while (self.get_tik() < nr_tiks) or picked <= (number_of_patients - 1):

            print("\n  TIK #" + str(self.get_tik()))

            if self.patient_counter < number_of_patients and self.get_tik() == self.patient_generation_list[self.patient_counter]:
                self.generate_patient((self.min_weight + self.max_weight)/2)
            
            # 1) ambulances inform each other of their metrics
            metrics = []                    
            for a in self.ambulances:
                metrics.append(a.send_metrics(self.g, self.patients, self.hospital))
                        
            # 2) ambulances decide which is the one executing
            for a in self.ambulances:
                a.collect_metrics_and_decide(metrics, self.patients, self.g, self.charging_stations)

            # 3) chosen ambulances execute
            for a in self.ambulances:
                o = a.execute(self.hospital.get_hospital_pos())
                if o == "picked_up_success":
                    self.success += 1
                    self.patients.remove(a.get_patient())
                    picked += 1
                elif o == "picked_up_failed":
                    self.failed += 1
                    self.patients.remove(a.get_patient())
                    picked += 1


            self.increment_tik()

        graph_utils.draw_graph(self.g)
        print(self.patients)
        print("\n\nThe simulation ended with " + str(self.success) + " cases of success and " + str(self.failed) + " cases of failure\n")
        #for a in self.ambulances:
        #    a.printQMatrix()
        


#---------------------------------------------------------------------------------
#------------------------------------ MAIN ---------------------------------------
#---------------------------------------------------------------------------------
class Main(object):

    def clear(self):
        print("\n"*40)

    def read_input(self, message, input_type): # input_type: "INT" | "Y/N"
        if (input_type == "INT"):
            try:
                return int(input(message))
            except Exception as e:
                self.clear()
                print("\nPlease enter a valid integer\n")
        elif (input_type == "Y/N"):
            while(True):
                user_input = input(message)
                if(user_input == 'Y' or user_input == 'y'):
                    return "y"
                elif(user_input == 'N' or user_input == 'n'):
                    return "n"
                else:
                    self.clear()
                    print("\nPlease enter a valid option (y/n)\n")
        elif input_type == "STR":
            try:
                return input(message)
            except Exception as e:
                self.clear()
                print("\nPlease enter a valid string\n")


    def metrics(self, emergency):
        self.clear()
        print("\n\n\n\n\n\n\n---------------------------------------------------------------------")
        print("\nMetrics - Choose your own metrics for the simulation")
        print("1 - Percentage of Each Priority in Patient Generation")
        print("2 - Time for Ambulance to Successfully Reaching Patient")
        print("3 - Number of Generated Patients During Exploration")
        print("0 - Exit\n")
        case = self.read_input("Option:  ", "INT")
        if case == 0:
            sys.exit()
        if case == 1:
            while(True):
                print("\n\nFor each of the three priority types (low, moderate and medium)")
                print("please select a percentage. Percentages must add up to 100.\n")
                low = self.read_input("Low:  ", "INT")
                moderate = self.read_input("Moderate:  ", "INT")
                high = self.read_input("High:  ", "INT")
                
                if((low + moderate + high) == 100): 
                    emergency.set_priority((low, moderate, high))
                    return
                else:
                    self.clear()
                    print("\nWarning: Percentages must add up to 100\n")
        if case == 2:
                print("\n\nFor each of the three priority types (low, moderate and medium)")
                print("please select the maximum time.\n")
                low = self.read_input("Low:  ", "INT")
                moderate = self.read_input("Moderate:  ", "INT")
                high = self.read_input("High:  ", "INT")
                emergency.set_priority_time((low, moderate, high))
        if case == 3:
                print("\n\nPlease select the number of patients to be generated during")
                print("the reinforcement learning's exploration phase.")
                print("Warning: Large values will take some time to process\n")
                n = self.read_input("Number of patients:  ", "INT")
                emergency.set_n_training_patients(n)


    def start(self):
        emergency = Emergency( )
        self.clear()
        print("\n\n\n\n\n\n\n---------------------------------------------------------------------")
        print("Emergency Responses Simulator")
        print("\nThis program was developed in the context of the Autonomous and")
        print(   "Multi-Agent Systems course at Instituto Superior Técnico (2020).")
        print("\nThis is a multi-agent system implementation responsible to respond")
        print(   "to emergency situations and analyse different approaches.")
        print("\nAmbulances are parked and depart from the hospital.")
        print("\nEnvironment for the simulation:")
        print("1 - Use a randomly generated environment")
        print("2 - Generate environment")
        print("0 - Exit\n")
        case = self.read_input("Option:  ", "INT")
        if case == 0:
            sys.exit()
        if case == 1:
            emergency.set_graph(emergency.build_graph())
        if case == 2:
            emergency.set_n_nodes(self.read_input("Number of nodes:  ", "INT"))
            emergency.set_min_weight(self.read_input("Minimum weight between two nodes:  ", "INT"))
            emergency.set_max_weight(self.read_input("Maximum weight between two nodes:  ", "INT"))
            emergency.set_graph(emergency.build_graph())
        
        self.clear()
        
        while True:
            print("\n\n\n\n\n\n\n---------------------------------------------------------------------")
            print("\nPick the type of simulation:")
            print("1 - Hospital assigns ambulances")
            print("2 - Ambulances cooperate with one another - with reinforcement learning")
            print("9 - Change metrics")
            print("0 - Exit\n")
            case = self.read_input("Option:  ", "INT")

            if case == 0:
                sys.exit()

            # ------------------------ HOSPITAL ----------------------------------------------------------
            elif case == 1:
                #--- costumize?
                costumize = self.read_input("Do you wish to customize your simulation? (y/n)", "Y/N")
                if(costumize == "y"):
                    n_tiks = self.read_input("Number of TIKs:  ", "INT")
                    emergency.set_nr_ambulances(self.read_input("Number of ambulances:  ", "INT"))
                    emergency.set_nr_charging_stations(self.read_input("Number of charging stations:  ", "INT"))
                    emergency.run_hospital_decide_simulation("NCA", n_tiks)
                else:
                    emergency.run_hospital_decide_simulation("NCA")
                plt.savefig("./images/graph.png") # saving this graph as graph.png

            # ------------------------ AMBULANCES --------------------------------------------------------
            elif case == 2:
                #--- costumize?
                costumize = self.read_input("Do you wish to customize your simulation? (y/n) ", "Y/N")
                if(costumize == "y"):
                    n_tiks = self.read_input("Number of TIKs:  ", "INT")
                    emergency.set_nr_ambulances(self.read_input("Number of ambulances:  ", "INT"))
                    emergency.set_nr_charging_stations(self.read_input("Number of charging stations:  ", "INT"))
                    emergency.set_action_selection(self.read_input("Action Selection: (eGreedy/softMax/selective) ", "STR"))
                    emergency.set_learning_approach(self.read_input("Learning Approach: (QLearning/SARSA) ", "STR"))
                    emergency.run_cooperative_ambulances_simulation(n_tiks)
                else:
                    emergency.run_cooperative_ambulances_simulation()
                plt.savefig("./images/graph.png") # saving this graph as graph.png

            elif case == 9: #Change Metrics
                self.metrics(emergency)
                self.clear()

            else:
                self.clear()
                print(f"\n{case} is not a valid type of emergency...\n")
                continue

main = Main()
main.start()
