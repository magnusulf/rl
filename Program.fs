open System
open ValueIteration
open QLearning
open GridWorld
open RLCore

let createMdp maximumX maximumY blockedStates absorbitionStates turnProbability stayProbability livingReward: MDP<State, Action> = 
    let mdp_x = GridWorld.getStates maximumX maximumY blockedStates
    let mdp_a = GridWorld.getActions
    let mdp_tf: RLCore.TransitionFunction<GridWorld.State, GridWorld.Action> =
        GridWorld.transitionfunction maximumX maximumY blockedStates absorbitionStates turnProbability stayProbability
    let mdp_rf: GridWorld.State->GridWorld.Action->GridWorld.State->GridWorld.Reward =
        GridWorld.rewardFunction absorbitionStates livingReward
    let mdp_si = GridWorld.stateIndex maximumX maximumY
    let mdp_ai = GridWorld.actionIndex
    {X = mdp_x ; A = mdp_a ; p = mdp_tf ; r = mdp_rf ; si = mdp_si ; ai = mdp_ai }   

let createEnv maximumX maximumY blockedStates absorbitionStates turnProbability stayProbability livingReward: Env<State, Action> = 
    let mdp_x = GridWorld.getStates maximumX maximumY blockedStates
    let mdp_a = GridWorld.getActions
    let mdp_tfs: QLearning.TransitionFunction<GridWorld.State, GridWorld.Action> =
        GridWorld.transitionfunctionStochastic maximumX maximumY blockedStates absorbitionStates turnProbability stayProbability
    let mdp_rf: GridWorld.State->GridWorld.Action->GridWorld.State->GridWorld.Reward =
        GridWorld.rewardFunction absorbitionStates livingReward
    let mdp_si = GridWorld.stateIndex maximumX maximumY
    let mdp_ai = GridWorld.actionIndex
    {X = mdp_x ; A = mdp_a ; absorbtionStates = [GridWorld.absorbtionMarker] ;
    p = mdp_tfs ; r = mdp_rf ; si = mdp_si ; ai = mdp_ai}    

let policyRandom (Q: float[,]) (s: GridWorld.State) : GridWorld.Action = LanguagePrimitives.EnumOfValue<int, GridWorld.Action>(RLCore.rand.NextInt64(4) |> int)
let policyEpsilonGreedy epsilon (si: StateIdx<State>) (Q: float[,]) (s: GridWorld.State) : GridWorld.Action =
        let rnd = RLCore.rand.NextDouble()
        if (rnd <= epsilon) then // Returner tilfældig action ind i mellem
            LanguagePrimitives.EnumOfValue<int, GridWorld.Action>(RLCore.rand.NextInt64(4) |> int)
        else
            let Qvalues = Q[si s, *]
            if (Array.sum Qvalues < 0.1) then // Random when we barely know something
                LanguagePrimitives.EnumOfValue<int, GridWorld.Action>(RLCore.rand.NextInt64(4) |> int)
            else
                let a = Qvalues |> Seq.mapi (fun i v -> i, v) |> Seq.maxBy snd |> fst // Få indeks af den action med størt forventede værdi
                LanguagePrimitives.EnumOfValue<int, GridWorld.Action>(a) // Konverter fra int til action

let runGridworld maximumX maximumY blockedStates absorbitionStates initialState turnProbability stayProbability livingReward discount : unit =
    let mdp = createMdp maximumX maximumY blockedStates absorbitionStates turnProbability stayProbability livingReward

    // let Q = ValueIteration.valueIteration mdp discount

    // printfn "ValueIteration"
    // GridWorld.printV maximumX maximumY Q
    // GridWorld.printActions maximumX maximumY Q blockedStates
    

    let env = createEnv maximumX maximumY blockedStates absorbitionStates turnProbability stayProbability livingReward

    // printfn "Q learning random"    
    // let Q1 = QLearning.qLearn env policyRandom initialState discount
    // GridWorld.printV maximumX maximumY Q1
    // GridWorld.printActions maximumX maximumY Q1 blockedStates

    printfn "Q learning epsilon greedy"
    let Q2 = QLearning.qLearn env (policyEpsilonGreedy 0.25 env.si) initialState discount
    GridWorld.printV maximumX maximumY Q2
    GridWorld.printActions maximumX maximumY Q2 blockedStates

let runGridworld1 (a: int) : unit =
    let maximumX = 4
    let maximumY = 3
    let blockedStates = [1,1]
    let absorbitionStates: Map<GridWorld.State, GridWorld.Reward> = Map [(3,2), 1.0 ; (3,1), -1.0]
    let turnProbability = 0.1
    let stayProbability = 0.0
    let livingReward = 0.0
    let discount = 0.9

    runGridworld maximumX maximumY blockedStates absorbitionStates (0,0) turnProbability stayProbability livingReward discount

let runGridworld2 (a: int) : unit =
    let maximumX = 5
    let maximumY = 5
    let blockedStates = [2,3 ; 1,2 ; 2,2 ; 2,0 ; 4,2]
    let absorbitionStates: Map<GridWorld.State, GridWorld.Reward> = Map [(4,0), 1.0]
    let turnProbability = 0.1
    let stayProbability = 0.1
    let livingReward = 0.0
    let discount = 0.95

    runGridworld maximumX maximumY blockedStates absorbitionStates (0,4) turnProbability stayProbability livingReward discount

let runGridworld3 (a: int) : unit =
    let maximumX = 8
    let maximumY = 7
    let blockedStates = [0,3 ; 1,3 ; 2,3 ; 3,3 ; 5,3 ; 6,3 ; 7,3]
    let absorbitionStates: Map<GridWorld.State, GridWorld.Reward> = Map [(7,0), 1.0]
    let turnProbability = 0.1
    let stayProbability = 0.1
    let livingReward = 0.0
    let discount = 0.95

    runGridworld maximumX maximumY blockedStates absorbitionStates (0,6) turnProbability stayProbability livingReward discount



runGridworld3 1