module QLearning
open System
open RLCore

type Reward = float
type TransitionFunction<'s, 'a> = 's->'a->'s
type Policy<'s, 'a> = 's->'a
type RewardFunction<'s, 'a> = 's->'a->'s->Reward
type Env<'s, 'a when 's : equality> = { X : 's list; absorbtionStates: 's list ; A : 'a list;
p : TransitionFunction<'s, 'a>; r : RewardFunction<'s, 'a> ;
si: StateIdx<'s> ; ai: ActionIdx<'a> }

/// Konverterer en BaseTransitionFunciton til en stokastisk transitionfunction til brug i Q-Learning
let convertTransitionFunctionStochastic<'s, 'a when 's : equality> (bt: BaseTransitionFunction<'s, 'a>) : TransitionFunction<'s, 'a> =
    let tf s1 a : 's =
        let probs = bt s1 a
        RLCore.getRandomElementWeighted (List.map fst probs) (List.map snd probs)
    tf

let calcRewardMatrix(env: Env<'s, 'a>) : float[,,] =
    let ret: float[,,] = Array3D.init (List.length env.X) (List.length env.A) (List.length env.X) (fun i j k -> 0.0)
    for s1 in env.X do
        for a in env.A do
            for s2 in env.X do
                ret[env.si s1, env.ai a, env.si s2] <- env.r s1 a s2
    ret

let qLearn<'s, 'a when 's : equality> (env: Env<'s, 'a>) (policy: Policy<'s, 'a>) (initialState: 's) (discount: float) : float[,] =
    let R = calcRewardMatrix env
    let getActionStateValue = getActionToStateValue env.si env.ai R discount

    let Q = Array2D.zeroCreate (List.length env.X) (List.length env.A)
    let mutable currentState = initialState
    let mutable learningRate = 0.2
    for i in 1..1000000 do
        if (List.contains currentState env.absorbtionStates) then
            currentState <- initialState
        if (i % 10000 = 0) then
            learningRate <- learningRate*0.9
        let a = policy currentState
        let nextState = env.p currentState a
        let reward = getActionStateValue currentState a nextState Q
        Q[env.si currentState, env.ai a] <- (1.0-learningRate) * Q[env.si currentState, env.ai a] + learningRate * reward
        currentState <- nextState

    Q