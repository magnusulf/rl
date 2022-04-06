open System
open ValueIteration
open QLearning
open RLCore

let createMdp maximumX maximumY blockedStates initialState (rewardStates: Map<GridWorld.State, GridWorld.Reward>) turnProbability stayProbability livingReward: MDP<GridWorld.State, GridWorld.Action> = 
    let mdp_x = GridWorld.getStates maximumX maximumY blockedStates
    let maxStateIdx = (List.length mdp_x) + (List.length blockedStates) - 1;
    let mdp_a = GridWorld.getActions
    let mdp_tf: RLCore.TransitionFunction<GridWorld.State, GridWorld.Action> =
        GridWorld.transitionfunction maximumX maximumY blockedStates initialState rewardStates turnProbability stayProbability
    let mdp_rf: GridWorld.State->GridWorld.Action->GridWorld.Reward =
        GridWorld.rewardFunction rewardStates livingReward
    let mdp_si = GridWorld.stateIndex maximumX maximumY
    let mdp_ai = GridWorld.actionIndex
    {X = mdp_x ; maxStateIdx = maxStateIdx ; A = mdp_a ; p = mdp_tf ; r = mdp_rf ; si = mdp_si ; ai = mdp_ai }   

let createEnv maximumX maximumY blockedStates initialState rewardStates turnProbability stayProbability livingReward: Env<GridWorld.State, GridWorld.Action> = 
    let mdp_x = GridWorld.getStates maximumX maximumY blockedStates
    let maxStateIdx = (List.length mdp_x) + (List.length blockedStates) - 1;
    let mdp_a = GridWorld.getActions
    let mdp_tfs: QLearning.TransitionFunction<GridWorld.State, GridWorld.Action> =
        GridWorld.transitionfunctionStochastic maximumX maximumY blockedStates initialState rewardStates turnProbability stayProbability
    let mdp_rf: GridWorld.State->GridWorld.Action->GridWorld.Reward =
        GridWorld.rewardFunction rewardStates livingReward
    let mdp_si = GridWorld.stateIndex maximumX maximumY
    let mdp_ai = GridWorld.actionIndex
    {X = mdp_x ; maxStateIdx = maxStateIdx; A = mdp_a ; absorptionStates = [GridWorld.absorptionMarker] ;
    p = mdp_tfs ; r = mdp_rf ; si = mdp_si ; ai = mdp_ai}    

let policyRandom<'s, 'a> (actions: 'a list) (Q: float[,]) (s: 's) : 'a =
    actions[RLCore.rand.NextInt64(List.length actions) |> int]

let policyEpsilonGreedy<'s, 'a> epsilon (actions: 'a list) (si: StateIdx<'s>) (Q: float[,]) (s: 's) : 'a =
        let rnd = RLCore.rand.NextDouble()
        let count = List.length actions
        if (rnd <= epsilon) then // Returner tilfældig action ind i mellem
            actions[RLCore.rand.NextInt64(count) |> int]
        else
            let Qvalues = Q[si s, *]
            if (Array.sum Qvalues < 0.1) then // Random when we barely know something
                actions[RLCore.rand.NextInt64(count) |> int]
            else
                let a = Qvalues |> Seq.mapi (fun i v -> i, v) |> Seq.maxBy snd |> fst // Get index of the action with largest expected value
                actions[a] // Converts from int to action

let runGridworld maximumX maximumY blockedStates absorptionStates initialState turnProbability stayProbability livingReward discount : unit =
    let mdp = createMdp maximumX maximumY blockedStates initialState absorptionStates turnProbability stayProbability livingReward

    let sw1 = System.Diagnostics.Stopwatch.StartNew()
    let Q = ValueIteration.valueIteration mdp discount

    printfn "ValueIteration"
    GridWorld.printV maximumX maximumY Q
    GridWorld.printActions maximumX maximumY Q blockedStates (absorptionStates |> Map.toList |> List.map fst)

    sw1.Stop()
    printfn "millis: %f" sw1.Elapsed.TotalMilliseconds
    

    // let env = createEnv maximumX maximumY blockedStates initialState absorptionStates turnProbability stayProbability livingReward

    // let sw2 = System.Diagnostics.Stopwatch.StartNew()
    // printfn "Q learning epsilon greedy"
    // let Q2 = QLearning.qLearn env (policyEpsilonGreedy 0.00 env.A env.si) initialState discount
    // GridWorld.printV maximumX maximumY Q2
    // GridWorld.printActions maximumX maximumY Q2 blockedStates (absorptionStates |> Map.toList |> List.map fst)
    
    // sw2.Stop()
    // printfn "millis: %f" sw2.Elapsed.TotalMilliseconds

let runRiverSwim size reverseProbability stayProbability livingReward rightReward leftReward discount : unit =
    let mdp_x = RiverSwim.getStates size
    printfn "%A" mdp_x
    let maxStateIdx = size // Minus one because it is one indexed and plus one for the absobtion state
    let mdp_a = RiverSwim.getActions
    let mdp_tf: RLCore.TransitionFunction<RiverSwim.State, RiverSwim.Action> =
        RiverSwim.transitionfunction size reverseProbability stayProbability
    let env_tfs: QLearning.TransitionFunction<RiverSwim.State, RiverSwim.Action> =
        RiverSwim.transitionfunctionStochastic size reverseProbability stayProbability
    let mdp_rf: RiverSwim.State->RiverSwim.Action->RiverSwim.Reward =
        RiverSwim.rewardFunction size livingReward rightReward leftReward
    let mdp_si = RiverSwim.stateIndex size
    let mdp_ai: RiverSwim.Action->int = RiverSwim.actionIndex
    let mdp = {X = mdp_x ; maxStateIdx = maxStateIdx ;A = mdp_a ; p = mdp_tf ; r = mdp_rf ; si = mdp_si ; ai = mdp_ai }
    let policy_eg =  (policyEpsilonGreedy 0.20 mdp_a mdp_si) // policyRandom mdp_a
    let policy_r = policyRandom mdp_a


    let Q = ValueIteration.valueIteration mdp discount

    printfn "ValueIteration"
    RiverSwim.printV size Q
    RiverSwim.printActions size Q

    // Q-learning
    let env = {X = mdp_x ; maxStateIdx = maxStateIdx; A = mdp_a ; absorptionStates = [] ;
    p = env_tfs ; r = mdp_rf ; si = mdp_si ; ai = mdp_ai}

    let sw2 = System.Diagnostics.Stopwatch.StartNew()
    printfn "Q learning random"
    let Q2 = QLearning.qLearn env policy_r 0 discount
    RiverSwim.printV size Q2
    RiverSwim.printActions size Q2
    
    sw2.Stop()
    printfn "millis: %f" sw2.Elapsed.TotalMilliseconds

    let sw3 = System.Diagnostics.Stopwatch.StartNew()
    printfn "Q learning epsilon-greedy"
    let Q3 = QLearning.qLearn env policy_eg 0 discount
    RiverSwim.printV size Q3
    RiverSwim.printActions size Q3
    
    sw3.Stop()
    printfn "millis: %f" sw2.Elapsed.TotalMilliseconds


let runGridworld1 (a: int) : unit =
    let maximumX = 4
    let maximumY = 3
    let blockedStates = [1,1]
    let absorptionStates: Map<GridWorld.State, GridWorld.Reward> = Map [(3,2), 1.0 ; (3,1), -1.0]
    let turnProbability = 0.1
    let stayProbability = 0.0
    let livingReward = 0.0
    let discount = 0.9

    runGridworld maximumX maximumY blockedStates absorptionStates (0,0) turnProbability stayProbability livingReward discount

let runGridworld2 (a: int) : unit =
    let maximumX = 5
    let maximumY = 5
    let blockedStates = [2,3 ; 1,2 ; 2,2 ; 2,0 ; 4,2]
    let absorptionStates: Map<GridWorld.State, GridWorld.Reward> = Map [(4,0), 1.0]
    let turnProbability = 0.1
    let stayProbability = 0.1
    let livingReward = 0.0
    let discount = 0.95

    runGridworld maximumX maximumY blockedStates absorptionStates (0,4) turnProbability stayProbability livingReward discount

let runGridworld3 (a: int) : unit =
    let maximumX = 8
    let maximumY = 7
    let blockedStates = [0,3 ; 1,3 ; 2,3 ; 3,3 ; 5,3 ; 6,3 ; 7,3]
    let absorptionStates: Map<GridWorld.State, GridWorld.Reward> = Map [(7,0), 1.0]
    let turnProbability = 0.0
    let stayProbability = 0.0
    let livingReward = 0.0
    let discount = 0.9

    runGridworld maximumX maximumY blockedStates absorptionStates (0,6) turnProbability stayProbability livingReward discount


// size reverseProbability stayProbability livingReward rightReward leftReward discount
//runRiverSwim 8 0.05 0.55 0 1 0.12 0.8

runGridworld3 1