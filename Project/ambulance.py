import graph_utils
import enum
import numpy as np
from random import random, randint
import enum
import numpy as np
from random import random, randint

class Ambulance:
	def __init__(self, id, pos, pos_list, en = 100, capacity = 1, hospital = None, graph = None, a_s=None, l_a=None):
		self.id = id				   #ambulance id
		self.totalCapacity = capacity  # number of patients the ambulance can accommodate
		self.pos = pos
		self.patient = None
		self.available = True		   #the ambulance is available at first, set to False if goal == pick_up_patient, go_to_hospital, get_energy"
		self.energy_available = en	   #the level of energy the ambulance has (X energy = X minute autonomy)
		self.batery_capacity = en	   #the total ammount of energy the ambulance can have
		self.goal = "no_goal_yet" 	   #type of goals: "no_goal_yet, pick_up_patient, go_to_hospital, go_to_energy, get_energy, head_position"
		self.goal_path = [-1,-1,-1,-1] #[X,Y,Z,W] X: ambulance->patient Y: patient->hospital Z: ambulance->charging_station W: origin -> destination
		self.charging_station_to_charge = None #used to charge
		self.tik = -1

		#for collaborative ambulances
		self.hospital = hospital
		
		#for reinforcement learning
		if graph is None:
			self.learning = False
			return
		self.learning = True
		self.g = graph
		self.curr_patients = list()
		self.action_selection = a_s
		self.learning_approach = l_a
		#self.action = pos_list	# Locations == Vertices
			# ['pickUpPacient', 'goToHospital'] - the agent does not need to learn from these actions
		self.it, self.total = 0, 100000
		#self.locations = pos_list
		self.discount_factor, self.learning_rate = 0.9, 0.5
		self.epsilon, self.rand_factor = 0.7, 0.05
		self.actions = pos_list
			# Possible actions correspond to the move to a given position
		self.q = np.zeros((pos_list, pos_list))	#creates the initial Q-value function structure:
		self.dec = (self.epsilon - 0.1) / self.total		#	(x y action) <- 0
		self.target = None		# Intended destination
		self.exploration = True

	#---------------------------------------------------------------------------------
	#  								Getters and Setters
	#---------------------------------------------------------------------------------

	def __repr__(self):
		return f"Ambulance {self.id} with energy {self.energy_available} has goal: {self.goal} "

	def get_goal(self): 
		return self.goal

	def set_goal_pacient(self, p, a_p, p_h): #(Pacient, TIKS from ambulance to patient, TIKS from patient to hospital)
		self.goal = "pick_up_patient"
		self.patient = p
		self.goal_path = [a_p, p_h, -1, -1]
		self.available = False

	def set_goal_charge(self, a_cs, target):
		self.goal = "go_to_energy"
		self.goal_path = [-1, -1, a_cs, -1]
		#eventualmente podemos precisar de saber onde fica essa station (target)
		self.charging_station_to_charge = target
		self.available = False

	def get_goal(self):
		return self.goal

	def get_id(self):
		return self.id

	def set_pos(self,pos):
		self.pos = pos

	def get_pos(self):
		return self.pos

	def set_patient(self, pat):
		self.patient= pat
	
	def get_patient(self):
		return self.patient

	def set_chosen_ambulance_id(self, id):
		self.chosen_ambulance_id= id

	def get_chosen_ambulance_id(self):
		return self.chosen_ambulance_id
	
	def get_energy_available(self):
		return self.energy_available
		
	def consume_energy(self):
		self.energy_available -= 1 # Each TIK consumes 1 unit of energy

	def get_batery_capacity(self):
		return self.batery_capacity

	def is_available(self):
		return self.available

	def increment_tik(self): #TIK N -- execute code of TIK N -- call incement_tik() to set TIK N+1 for the next iteration
		self.tik += 1

	def set_exploitation(self):
		self.exploration = False

	#---------------------------------------------------------------------------------
    #                               Execute
    #---------------------------------------------------------------------------------

	def execute(self, hospital_pos = None):
		self.increment_tik()

		if self.goal == "pick_up_patient":
			if (self.goal_path[0] == 0): # picking up patient
				self.goal = "go_to_hospital"
				#self.pos = self.get_patient().pos

				print("Ambulance #" + str(self.get_id()) + " just picked up patient #" + str(self.get_patient().get_id()) + ". Battery %" + str(self.get_energy_available()) )

				time_to_arrive = self.get_patient().get_time_of_occurrence() + self.get_patient().get_emergency_time() - self.tik
				if(time_to_arrive >= 0):
					print("Ambulance #" + str(self.get_id()) + " status: SUCCESS")
					return "picked_up_success" #send status to simulation for patient to be removed
				else:
					print("Ambulance #" + str(self.get_id()) + " status: FAILED (by " + str(abs(time_to_arrive)) + " TIK)")
					return "picked_up_failed" #send status to simulation for patient to be removed
			else:
				self.goal_path[0] -= 1
				self.consume_energy()

		elif self.goal == "go_to_hospital":
			if (self.goal_path[1] == 1): # left patient at the hospital
				print("Ambulance #" + str(self.get_id()) + " left patient #" + str(self.get_patient().get_id()) + " at the hospital" + ". Battery %" + str(self.get_energy_available()) )
				
				if self.energy_available < self.batery_capacity * 0.50: #ambulance can charge in the hospital
					self.goal = "get_energy"
					self.charging_station_to_charge = self.hospital.get_charging_station()
					
				else:
					self.goal = "no_goal_yet"
					self.available = True
					self.pos = hospital_pos

			else:
				self.goal_path[1] -= 1
				self.consume_energy()

		elif self.goal == "go_to_energy":
			if (self.goal_path[2] == 0): # is now in the charging station
				print("Ambulance #" + str(self.get_id()) + " is now in a charging station. Battery %" + str(self.get_energy_available()) )
				self.goal = "get_energy"
				self.charging_station_to_charge.start_charge()
				self.pos = self.charging_station_to_charge.get_pos()
			else:
				self.goal_path[2] -= 1
				self.consume_energy()

		elif self.goal == "get_energy": 
			av = self.charging_station_to_charge.recharge_energy(self.id, self)
			if (av):
				self.available = True
				self.charging_station_to_charge.stop_charge()
				self.goal = "no_goal_yet"

		elif self.learning:
			if self.goal == "head_position":
				# the agent is moving through an edge
				if self.original_action is not None:
					r = self.reward(self.original_state, self.original_action)
					if r > 0:
						self.learningDecision(self.original_state, self.original_action, r)
						self.original_action, self.original_state = None, None

				self.goal_path[3] -= 1
				if self.goal_path[3] == 0 or self.goal_path[3] == -1:  # reached
					self.goal = "no_goal_yet"
					self.pos = self.target

			elif self.goal == "no_goal_yet":
				self.original_state = self.getState()
				self.original_action = self.selectAction()
				self.executeQ(self.original_action)
				r = self.reward(self.original_state, self.original_action)
				if r > 0:
					self.learningDecision(self.original_state, self.original_action, r)
					self.original_action, self.original_state = None, None


	#---------------------------------------------------------------------------------
	#                               Collaborative behavior
	#---------------------------------------------------------------------------------

	def send_metrics(self, g, patients, hospital):  #sends lst [ (d_to_pat. d_to_hospital,has enough energy, is available),.... ] indexed by patient
		metrics_per_pat = []
		
		self.curr_patients.clear()
		self.curr_patients = patients.copy()

		for p in patients:
			d_to_pat = graph_utils.shortest_path(g, self.get_pos(), p.get_patient_pos())[0]
			d_to_h = graph_utils.shortest_path(g, hospital.get_hospital_pos(), p.get_patient_pos())[0]
			if(self.energy_available >= d_to_pat + d_to_h):
				has_enough_energy = True
			else:
				has_enough_energy = False

			metrics_per_pat.append([d_to_pat, d_to_h, has_enough_energy, self.is_available()])
		return metrics_per_pat

	def collect_metrics_and_decide(self,metrics, patients,g,charging_stations):
		#metrics = [[(12,1, T, T), (11,3, F, T)], [(8,4, T, T), (6, 3,F, T)]]
		#   		  		    [Amb1],		                [Amb2]
		#	      			[(pat1), (pat2)]        	[(pat1), (pat2)]
		
		for p in range(len(patients)):
			if(not patients[p].is_assigned()):
				best_id = -1
				best_metrics = 100000**8
				best_path_to_p = -1
				best_path_to_h = -1
				for a in range(len(metrics)):
					if (metrics[a][p][3] == True) and (metrics[a][p][2] == True) and ((metrics[a][p][1] + metrics[a][p][0]) < best_metrics):
						best_id = a
						best_metrics = metrics[a][p][0] + metrics[a][p][1]
						best_path_to_p = metrics[a][p][0]
						best_path_to_h = metrics[a][p][1]
					
				if self.id == best_id:
					self.set_goal_pacient(patients[p], best_path_to_p, best_path_to_h)
					patients[p].set_assigned(True)
					i=0
					while i < len(patients):
						metrics[self.id][i][3] = False #set availability to False
						i += 1
		
		if self.available and self.energy_available < (self.batery_capacity * 0.5) :
			(best_distance, best_cs) = graph_utils.closest_charging_station(g, self.pos, charging_stations)
			self.set_goal_charge(best_distance, best_cs)

	#---------------------------------------------------------------------------------
	#                               Reinforcement Learning
	#---------------------------------------------------------------------------------

	class ActionSelection(enum.Enum):
		eGreedy = 1
		softMax = 2
		selective = 3

	class LearningApproach(enum.Enum):
		QLearning = 1
		SARSA = 2

	def getLearningMetrics(self, g, patients, hospital):
		self.curr_patients.clear()
		self.curr_patients = patients.copy()

	def executeLearning(self):
		if self.goal == "head_position":
			# the agent is moving through an edge
			self.goal_path[3] -= 1
			if self.goal_path[3] == 0 or self.goal_path[3] == -1:  # reached
				self.goal = "no_goal_yet"
				self.pos = self.target

		elif self.goal == "no_goal_yet":
			self.original_state = self.getState()
			self.original_action = self.selectAction()
			self.executeQ(self.original_action)
			r = self.reward(self.original_state, self.original_action)
			self.learningDecision(self.original_state, self.original_action, r)
			#self.original_action, self.original_state = None, None

	def getState(self):
		#Accesses the state of an agent given its position
		#availability = 2 if self.available else 1
		# * self.locations + availability * self.locations
		return self.get_pos()

	def learningDecision(self, original_state, original_action, u):
		#Learns a policy up to a certain step and then uses policy to behave
		self.it += 1
		previous_q = self.getQ(original_state, original_action)

		self.epsilon = max(self.epsilon-self.dec, 0.05)
		# percept = self.nextPos()

		if self.learning_approach == Ambulance.LearningApproach.QLearning:
			pred_error = u + self.discount_factor * self.getMaxQ(self.getState()) - previous_q
		else:
			new_action = self.selectAction()
			pred_error = u + self.discount_factor * self.getQ(self.getState(), new_action) - previous_q

		self.setQ(original_state, original_action, previous_q + self.learning_rate * pred_error)
		###if self.it % 1000 == 0:
			###print("\tepsilon = " + str(self.epsilon) + "\n")

	def executeQ(self, action = None):
		#Executes action according to the learned policy
		if action is None:
			pass

		else:
			self.goal = "head_position"
			self.goal_path[3] = graph_utils.shortest_path(self.g, self.get_pos(), action)[0]
			self.target = action


	def selectAction(self):
		#Selects action according to e-greedy or soft-max strategies
		self.epsilon -= self.dec
		#if random() < self.rand_factor:
		#	return self.randomAction()
		if self.action_selection == Ambulance.ActionSelection.selective:
			return self.selective()
		elif self.action_selection == Ambulance.ActionSelection.eGreedy:
			return self.eGreedy()
		elif self.action_selection == Ambulance.ActionSelection.softMax:
			return self.softMax()

	def randomAction(self):
		#Selects a random action
		valid_actions = self.availableActions()
		return valid_actions[randint(0, len(valid_actions)-1)]

	def eGreedy(self):
		#eGreedy action selection
		valid_actions = self.availableActions()
		if random() > self.epsilon:
			return valid_actions[randint(0, len(valid_actions)-1)]
		return self.getMaxActionQ(self.getState(), valid_actions)

	def softMax(self):
		#SoftMax action selection
		valid_actions = self.availableActions()
		cumulative = np.zeros(len(valid_actions))

		cumulative[0] = np.exp(self.getQ(self.getState(), 0) / self.epsilon * 100.0)
		for i in range(1, len(valid_actions)):
			cumulative[i] = np.exp(self.getQ(self.getState(), i) / self.epsilon * 100.0) + cumulative[i - 1]

		total = cumulative[len(valid_actions) - 1]
		cut = random() * total

		for i in range(len(valid_actions)):
			if cut <= cumulative[i]:
				return valid_actions[i]
		return None

	def selective(self):
		if self.exploration:
			return self.randomAction()
		else:
			valid_actions = self.availableActions()
			return self.getMaxActionQ(self.getState(), valid_actions)

	def reward(self, state, action):
		#Retrieves reward from state
		reward = 0
		for p in self.curr_patients:
			d_to_pat = graph_utils.shortest_path(self.g, state, p.get_patient_pos())[0]
			if d_to_pat > 25:
				continue	# We disregard if the patient is too far
			reward += p.get_emergency_priority() * np.power(0.75, d_to_pat)
				# Distance to the patient inversely proportional to the reward
		return reward

	def getMaxActionQ(self, state, actions_indexes):
		#Gets the index of maximum Q-value action for a state
		maximum = -np.Inf
		max_index = -1
		for i in actions_indexes:
			v = self.q[state][i]
			if v > maximum:
				maximum = v
				max_index = i
		return max_index

	def getMaxQ(self, state):
		#Gets the maximum Q-value action for a state (x y)
		return np.max(self.q[state])

	def getQ(self, state, action):
		#Gets the Q-value for a specific state-action pair (x y action)
		return self.q[state][action]

	def setQ(self, state, action, val):
		#Sets the Q-value for a specific state-action pair (x y action)
		self.q[state][action] = val

	def availableActions(self):
		#Returns the eligible actions- drive to the neighbors
		return list(self.g.adj[self.pos])

	def printQMatrix(self):
		print(self.q)
		print("\n")