module RLCore

open System

type StateIdx<'s> = 's->int
type ActionIdx<'a> = 'a->int
type BaseTransitionFunction<'s, 'a> = 's->'a->(('s * float) list)
type TransitionFunction<'s, 'a>= 's->'a->'s->float

let rand = Random()

let rec getRandomElementWeightedSub (lst: 'a list) (weigths: float list) (r: float): 'a =
    if (r - (List.head weigths)) > 0 then
        getRandomElementWeightedSub (List.tail lst) (List.tail weigths) (r - List.head weigths)
    else
        List.head lst

let getRandomElementWeighted (lst: 'a list) (weigths: float list): 'a =
    getRandomElementWeightedSub lst weigths (rand.NextDouble() * (List.sum weigths))

/// Konverterer en BaseTransitionFunciton til en almindelig transitionfunction til brug i value iteration
let convertTransitionFunction<'s, 'a when 's : equality> (bt: BaseTransitionFunction<'s, 'a>) : TransitionFunction<'s, 'a> =
    let tf (s1: 's) (a: 'a) (s2: 's) : float =
        let probs: ('s * float) list = bt s1 a
        let filtered = List.filter (fun x -> fst x = s2) probs
        match filtered with
        | n :: rest -> snd n
        | [] -> 0
    tf

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