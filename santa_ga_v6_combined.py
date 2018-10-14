import random
import array
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

from haversine import haversine
from deap import algorithms
from deap import base
from deap import creator
from deap import tools


data = pd.read_csv("./dataset from Kaggle/gifts.csv")

# Split Antartica and Non-Antartica
data_antartica = data[data["Latitude"] <= -60]
data_xantartica = data[data["Latitude"] > -60]

# Sort gifts by longitude
xantartica = data_xantartica.copy()
xantartica = xantartica.sort_values("Longitude")

antartica = data_antartica.copy()
antartica = antartica.sort_values("Longitude")

######## Assign trip number to gifts - Non-Antartica
wt = 0.0
trip_df = []
trip_no = 0
for i in range(len(xantartica)): 
    wt += xantartica.iloc[i,3]
    if wt <= 1000: 
        trip_df.append(trip_no)
    else: 
        wt = xantartica.iloc[i,3]
        trip_no += 1
        trip_df.append(trip_no)
        

trip_df = pd.DataFrame(trip_df)
xantartica = xantartica.reset_index(drop=True)
xantartica = xantartica.join(trip_df)
xantartica.rename(columns={xantartica.columns[4]:'trip'}, inplace=True)
xantartica["trip"].max() #1268


######## Assign trip number to gifts - Antartica
wt = 0.0
trip_df = []
trip_no = xantartica["trip"].max()+1
for i in range(len(antartica)): 
    wt += antartica.iloc[i,3]
    if wt <= 1000: 
        trip_df.append(trip_no)
    else: 
        wt = antartica.iloc[i,3]
        trip_no += 1
        trip_df.append(trip_no)
        
trip_df = pd.DataFrame(trip_df)
antartica = antartica.reset_index(drop=True)
antartica = antartica.join(trip_df)
antartica.rename(columns={antartica.columns[4]:'trip'}, inplace=True)
#antartica["trip"].max()-xantartica["trip"].max() #164
#antartica["trip"].max() #1431 total trips

######## Combine both Non-Antartica and Antartica

all = pd.concat([xantartica, antartica])

######## Split each trip into subarray

diff = np.diff(np.array(all)[:,4])
split = np.where(diff == 1)[0] + 1
all_bytrip = np.split(np.array(all), split)
#all_bytrip[0]

(all["trip"].max())


####### GA #############################################

total_wrw = 0.0
final_list = pd.DataFrame()
logbook = []
population_log = []

for i in range(int(all["trip"].max())+1):  
	trip = all_bytrip[i]
	TRIP_SIZE = len(trip)
	
	creator.create("FitnessMin", base.Fitness, weights = (-1.0,))
	creator.create("Individual", list, fitness = creator.FitnessMin)
	
	toolbox = base.Toolbox() 
	toolbox.register("indices", random.sample, range(TRIP_SIZE), TRIP_SIZE)
	toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
	toolbox.register("population", tools.initRepeat, list, toolbox.individual)

	def weariness(individual):
		newtrip = trip[individual] 
		weight = sum(newtrip[:,3])
		start_distance = haversine((90,0), newtrip[0, [1,2]])
		wrw = (weight + 10) * start_distance
		for i in range(1, len(individual)):
			weight -= newtrip[i-1, 3]
			wrw += (weight + 10) * haversine(newtrip[i-1, [1,2]], newtrip[i, [1,2]])
		end_distance = haversine(newtrip[-1, [1,2]], (90,0)) 
		wrw += 10 * end_distance
		return wrw,

	toolbox.register("mate", tools.cxPartialyMatched)
	toolbox.register("mutate", tools.mutShuffleIndexes, indpb = 0.05)
	toolbox.register("select", tools.selTournament, tournsize = 3)
	toolbox.register("evaluate", weariness)

	random.seed(169)
	pop = toolbox.population(n=300)
	hof = tools.HallOfFame(1)
	stats = tools.Statistics(lambda ind:ind.fitness.values)
	stats.register("min", np.min)
       
    #ngen = 200 with population 300 is about 45 seconds each trip
    #ngen = 500 with population 300 is about 90 seconds each trip
	pop, log = algorithms.eaSimple(pop, toolbox, cxpb = 0.7, mutpb = 0.2, ngen = 500, stats = stats, halloffame = hof, verbose = True)
    
	best = tools.selBest(pop,1)[0]
	(wrw, ) = best.fitness.values 
	
	print(f"best from trip {i} {best,best.fitness.values}")
    
    #save the log
	logbook.append(log)
	population_log.append(pop)
	
	# Get the GiftId
	best_trip = trip[best]
	best_trip = pd.DataFrame(best_trip[:, 0]).astype(int)
	best_trip[1] = i

	final_list = final_list.append(best_trip) 
	total_wrw += wrw
	print(f"Total Weariness after {i} trips is {total_wrw}")


final_list.columns = ["GiftId", "TripId"]
final_list.to_csv("final_list_test.csv", index = False)

total_wrw 

#save logbook
with open("logbook.txt", "w") as file:
    file.write(str(logbook))

with open("population.txt", "w") as file:
    file.write(str(population_log))


# wrw with ngen 5 = 94,217,213,567.58084
