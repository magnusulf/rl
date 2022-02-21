module QLearning

open RLCore

type Reward = float
type TransitionFunction<'s, 'a> = 's->'a->'s
type Policy<'s, 'a> = 's->'a
type RewardFunction<'s, 'a> = 's->'a->'s->Reward
type Env<'s, 'a> = { X : 's list; A : 'a list; p : TransitionFunction<'s, 'a>; r : RewardFunction<'s, 'a> ; si: StateIdx<'s> ; ai: ActionIdx<'a> }

let calcRewardMatrix(env: Env<'s, 'a>) : float[,,] =
    let ret: float[,,] = Array3D.init (List.length env.X) (List.length env.A) (List.length env.X) (fun i j k -> 0.0)
    for s1 in env.X do
        for a in env.A do
            for s2 in env.X do
                ret[env.si s1, env.ai a, env.si s2] <- env.r s1 a s2
    ret

let qLearn<'s, 'a> (env: Env<'s, 'a>) (policy: Policy<'s, 'a>) (initialState: 's) (discount: float) : float[,] =
    let R = calcRewardMatrix env
    let getActionStateValue = getActionToStateValue env.si env.ai R discount

    let Q = Array2D.zeroCreate (List.length env.X) (List.length env.A)
    let mutable currentState = initialState

    for _ in 1..100 do
        let a = policy currentState
        let nextState = env.p currentState a

        0.0 |> ignore

    Q