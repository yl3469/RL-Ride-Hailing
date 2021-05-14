# READ ME

We implement the sequential-decision-making Markov Decision Process proposed by Feng, Gluzman, and Dai (2020).

The global parameters include:
- horizon (H=360): the total number of epochs in a "working day"; this is divided evenly into three periods each with length 120.
- number of regions (R=5)
- number of cars (N=1000)
- passenger arrival rates (lambda)
- probability distribution over ride requests (P)
- travel times (tau): three-dimensional array, with tau[q,i,j] representing the travel time from region i to region j in period q.
  - we denote tau_max[q,j] as the maximum travel time to destination j in period q starting from any origin i.
- patience time (L): the maximum time a passenger is willing to wait for a matching.

The state has the following components:
- epoch: indicates the current epoch; takes values from 0 to H;
- cars: an iterated two-dimensional array; the [i,j] entry corresponds to the number of cars that have region i as their destination and are at distance/travel time j from region i;
- passengers: a two-dimensional array; the [i,j] entry is the number of ride requests from region i to region j;
- "do-nothing" cars: same shape as the cars component; records the number of cars that do not undergo a change in destination in the decision process for the current epoch. We will elaborate on this in our discussion of the transition dynamics later.


The action space is the set of all two-dimensional tuples each taking values in {0, ..., R-1}. Each action represents a trip assignment of the form (origin, destination) applied to a certain car. For each origin i, only a subset of **available cars** are considered for the action, namely the cars that have i as their destination and are within distance (tau_max[q,i]+L) of i. This upper bound represents the fact that passengers are only willing to wait up to L for a matching.

An action (trip assignment to a car) can fall into one of the three categories:
- matching of a ride request: it is assumed that passenger matching is always prioritized; that is, if there is a request from region o to region d, and there is a car available for trips starting from region o, this car is matched to the request. 
- empty car rerouting: if a car is idling at its current location and the action relocates it to another region.
- "do-nothing": the action does not change the destination of the car.

Note that at the beginning of each epoch, one can calculate the number of available cars for each origin i (i.e., heading toward region i and within patience range L) ; summing this number over all possible origins gives us the total number of available cars at the start of each epoch. Assigning trips to all these available cars together is a hard problem. However, the sequential decision-making (SDM) process at each epoch deconstructs the problem into a sequence of single trip assignments to each available car. More specifically, for each epoch, we loop over each origin and available car for that origin.
- If are available cars for the origin:
  - If there is a passenger ride request starting from the origin
    - we match an available car to the ride request and update the state components (cars, passengers) accordingly.
  - If there isn't a ride request starting from the origin
    - we decide a destination for the car
      - if the car is idling right at the origin and we decide to relocate it to another destination, this is an empty-car rerouting. We update the cars component and do not change the passenger component.
      - otherwise, we do not make any changes to the car (i.e., continue to have it heading toward the current origin). However, it is considered unavailable for the rest of the same SDM; therefore, we move them from the cars component to the "do-nothing" component.

Once an SDM finishes, the state transitions to a new epoch. Cars move forward one step towards their destinations. Unfulfilled ride requests are counted and the passenger matrix is reset to zero; then, new passenger arrivals are sampled from corresponding Poisson distributions.

The simulation does not explicitly implement the hierarchy structure of the epoch transitions vs. the transitions within the same SDM. We merely keep track of whether the current SDM is finished. If not (i.e., if we are in the middle of an SDM), the state transition only entails the change made to one car as determined by the action taken. Otherwise (i.e., the current SDM is finished), in addition to making the change to one car (the last car in the SDM), we also make the "global" changes, including incrementing the epoch by 1, moving the cars one step forward towards their destinations, and resetting the passenger component and sampling new passenger arrivals.

The reward function measures the number of fulfilled ride requests. Matched requests generate a reward of 1, while an empty rerouting and a "do-nothing" trip both generate a reward of 0. As we also keep track of the cumulative number of unfulfilled ride requests (by summing the remaining terms of the passenger matrix at the end of the SDM for each epoch), this allows us to calculate the fraction of fulfilled ride requests after the final epoch.
