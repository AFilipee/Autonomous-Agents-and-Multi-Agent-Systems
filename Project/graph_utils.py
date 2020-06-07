import networkx as nx 
from random import randint
import random
import matplotlib.pylab as plt

#colormap contains the color associated with each node
colormap = []

entity_color = {
  "default"         : "#c2c5cc", #light grey
  "hospital"        : "#96dcfa", #blue
  "ambulance"       : "#f5d0a9", #light orange
  "patient"         : "#a62308", #wine red
  "charging_station": "#8ade8c"  #light green
}

def generate_weighted_binomial_graph(n=15, p=0.2, min_weight=3, max_weight=15):
    #generate graph
    nok = True
    while(nok): #prevents zero degree nodes
        g = nx.binomial_graph(n, p)
        d = dict(nx.degree(g))
        if (0 not in d.values() and nx.is_connected(g)):
            nok = False

    #generate weights
    for (u,v,w) in g.edges(data=True):
        w['weight'] = randint(min_weight, max_weight)
    
    #initialize colormap
    global colormap 
    colormap = [entity_color.get("default")] * n

    return g

def draw_graph(graph):
    d = dict(nx.degree(graph))
    pos = nx.spring_layout(graph)
    plt.figure()
    nx.draw(graph, pos=pos, with_labels=graph.nodes().values(), nodelist=d.keys(), node_size=[(v+1) * 100 for v in d.values()], node_color=colormap)
    labels = nx.get_edge_attributes(graph,'weight')
    nx.draw_networkx_edge_labels(graph,pos, edge_labels=labels)

# Por enquanto o hospital é apenas a cor azul no mapa
def set_hospital(hospital_position):
    colormap[hospital_position] = entity_color.get("hospital")

# Por enquanto a ambulância é apenas a cor laranja no mapa
def set_ambulance(ambulance_nr):
    colormap[ambulance_nr] = entity_color.get("ambulance")

# Por enquanto o paciente é apenas ...
def set_patient(patient_pos_nr):
    colormap[patient_pos_nr] = entity_color.get("patient")

def set_charging_station(cs_nr):
    colormap[cs_nr] = entity_color.get("charging_station")

def get_colormap():
    return colormap

# Calculates the shortest path between a source and a target
# using Djikstra's algorithm. The outputs is a tuple where:
# 1st element -> is the total weight of the path
# 2nd element -> is a list of the sequence of nodes in the path
def shortest_path(g, source, target):
    return nx.single_source_dijkstra(g, source, target)

def closest_charging_station(g, source, stations):
    best_cs = stations[0]
    best_distance = float("inf")
    for cs in stations:
        if cs.has_enough_capacity():
            d = shortest_path(g, source, cs.get_charging_station_pos())[0]
            if d < best_distance:
                best_cs = cs
                best_distance = d
    return best_distance, best_cs

