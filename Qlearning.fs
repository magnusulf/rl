module QLearning
open System
open RLCore

type Reward = float
type Q = Reward[,]
type TransitionFunction<'s, 'a> = 's->'a->'s
type Policy<'s, 'a> = Q->'s->'a
type RewardFunction<'s, 'a> = 's->'a->Reward
type Env<'s, 'a when 's : equality> = { X : 's list; maxStateIdx: int; absorptionStates: 's list ; A : 'a list;
p : TransitionFunction<'s, 'a>; r : RewardFunction<'s, 'a> ;
si: StateIdx<'s> ; ai: ActionIdx<'a> }

/// Konverterer en BaseTransitionFunciton til en stokastisk transitionfunction til brug i Q-Learning
let convertTransitionFunctionStochastic<'s, 'a when 's : equality> (bt: BaseTransitionFunction<'s, 'a>) : TransitionFunction<'s, 'a> =
    let tf s1 a : 's =
        let probs = bt s1 a
        RLCore.getRandomElementWeighted (List.map fst probs) (List.map snd probs)
    tf

let calcRewardMatrix(env: Env<'s, 'a>) : float[,] =
    let ret: float[,] = Array2D.init (env.maxStateIdx+1) (List.length env.A) (fun i j -> 0.0)
    for s1 in env.X do
        for a in env.A do
                ret[env.si s1, env.ai a] <- env.r s1 a
    ret

let qLearn<'s, 'a when 's : equality> (env: Env<'s, 'a>) (policy: Policy<'s, 'a>) (initialState: 's) (discount: float) : float[,] =
    let R = calcRewardMatrix env
    let getActionStateValue = getActionToStateValue env.si env.ai R discount
    let maxQ = 1.0/(1.0-discount)
    let Q = Array2D.init (env.maxStateIdx+1) (List.length env.A) (fun i j -> maxQ)
    let visitCount = Array2D.zeroCreate (env.maxStateIdx+1) (List.length env.A)
    let mutable currentState = initialState
    let mutable accReward = 0.0
    for i in 1..1_000_000 do
        if (List.contains currentState env.absorptionStates) then
            currentState <- initialState
        
        let a = policy Q currentState
        let nextState = env.p currentState a
        let reward = getActionStateValue currentState a nextState Q
        let transitionReward = R[env.si currentState, env.ai a]
        visitCount[env.si currentState, env.ai a] <- visitCount[env.si currentState, env.ai a] + 1
        let k = 5.0
        let learningRate: float = k /(k + (visitCount[env.si currentState, env.ai a] |> float))
        accReward <- accReward + transitionReward
        Q[env.si currentState, env.ai a] <- (1.0-learningRate) * Q[env.si currentState, env.ai a] + learningRate * reward
        currentState <- nextState
    printfn "Accumulated reward: %.1f" accReward
    Q
