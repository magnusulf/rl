module RLCore

open System

type StateIdx<'s> = 's->int
type ActionIdx<'a> = 'a->int

let rand = Random()

let rec getRandomElementWeightedSub (lst: 'a list) (weigths: float list) (r: float): 'a =
    if (r - (List.head weigths)) > 0 then
        getRandomElementWeightedSub (List.tail lst) (List.tail weigths) (r - List.head weigths)
    else
        List.head lst

let getRandomElementWeighted (lst: 'a list) (weigths: float list): 'a =
    getRandomElementWeightedSub lst weigths (rand.NextDouble() * (List.sum weigths))


// Dealing with Q-values

/// Given a state and the calculated Q-table get the value of the state
/// Which is the action giving the maximum value
let getStateValue<'s> (si: StateIdx<'s>) (s: 's) (Q: float[,]) : float =
    Array.max Q[si s, *]

/// Given a starting state an action and the end state what is the value
/// This is equal to the transition reward plus the discounted (expected) value of the end state
let getActionToStateValue<'s, 'a> (si: StateIdx<'s>) (ai: ActionIdx<'a>) (R: float[,,]) (discount: float) (stateFrom: 's) (action: 'a) (stateTo: 's) (Q: float[,]) : float =
    let transitionReward = R[si stateFrom, ai action, si stateTo]
    let nextStateValue = getStateValue si stateTo Q
    transitionReward + discount * nextStateValue