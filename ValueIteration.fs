module ValueIteration

type StateIdx<'s> = 's->int
type ActionIdx<'a> = 'a->int

type Reward = float
type TransitionFunction<'s, 'a> = 's->'a->'s->float
type RewardFunction<'s, 'a> = 's->'a->'s->Reward
type MDP<'s, 'a> = { X : 's list; A : 'a list; p : TransitionFunction<'s, 'a>; r : RewardFunction<'s, 'a> ; si: StateIdx<'s> ; ai: ActionIdx<'a> }

/// Calculates a matrix that stores the transition probabilities
/// We store it so it needs not be calculated often
let getTransitionMatrix(mdp: MDP<'s, 'a>) : float[,,] =
    let ret: float[,,] = Array3D.init (List.length mdp.X) (List.length mdp.A) (List.length mdp.X) (fun i j k -> 0.0)
    for s1 in mdp.X do
        for a in mdp.A do
            for s2 in mdp.X do
                ret[mdp.si s1, mdp.ai a, mdp.si s2] <- mdp.p s1 a s2
    ret

/// Calculates a matrix that stores the rewards
/// We store it so it needs not be calculated often
let getRewardMatrix(mdp: MDP<'s, 'a>) : float[,,] =
    let ret: float[,,] = Array3D.init (List.length mdp.X) (List.length mdp.A) (List.length mdp.X) (fun i j k -> 0.0)
    for s1 in mdp.X do
        for a in mdp.A do
            for s2 in mdp.X do
                ret[mdp.si s1, mdp.ai a, mdp.si s2] <- mdp.r s1 a s2
    ret

/// Given a state and the calculated Q-table get the value of the state
/// Absorbtion states are those where 'si s < 0'
let getStateValue<'s> (si: StateIdx<'s>) (s: 's) (Q: float[,]) : float =
    Array.max Q[si s, *]

/// Given a starting state an action and the end state what is the value
/// This is equal to the transition reward plus the discounted (expected) value of the end state
let getActionToStateValue (discount: float) (si: StateIdx<'s>) (ai: ActionIdx<'a>) (stateFrom: 's) (action: 'a) (stateTo: 's) (R: float[,,])  (Q: float[,]) : float =
    let transitionReward = R[si stateFrom, ai action, si stateTo]
    let nextStateValue = getStateValue si stateTo Q
    transitionReward + discount * nextStateValue

/// Given a starting state and an action it gives the discounted, expected value
/// Of performing the action
let getActionValue (discount: float) (mdp: MDP<'s, 'a>) (stateFrom: 's) (action: 'a) (P: float[,,]) (R: float[,,])  (Q: float[,]) : float =
    let prob stateTo = P[mdp.si stateFrom, mdp.ai action, mdp.si stateTo]
    let value stateTo = getActionToStateValue discount mdp.si mdp.ai stateFrom action stateTo R Q
    let probValue stateTo = (prob stateTo) * (value stateTo)
    List.sum (List.map probValue mdp.X)

let valueIteration<'s, 'a> (mdp: MDP<'s, 'a>) (discount: float): float[,] =
    let P = getTransitionMatrix mdp
    let R = getRewardMatrix mdp
    
    let Q: float[,] = Array2D.init (List.length mdp.X) (List.length mdp.A) (fun i j -> 0.0)
    let mutable changes = 1
    while changes <> 0 do
        printfn "Iterating"
        changes <- 0
        for x in mdp.X do
                for a in mdp.A do
                    let newVal = getActionValue discount mdp x a P R Q
                    let oldVal = Q[mdp.si x, mdp.ai a]
                    if (newVal <> oldVal) then 
                        changes <- changes + 1
                    Q[mdp.si x, mdp.ai a] <-  newVal
    Q
