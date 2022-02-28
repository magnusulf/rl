open System
open ValueIteration
open QLearning
open RLCore

// For more information see https://aka.ms/fsharp-console-apps
//type MDP = { X : State list; A : Action list; p : TransitionFunction; r : RewardFunction }

let maximumX = 4
let maximumY = 3
let blockedStates = [1,1]
let absorbitionStates: Map<GridWorld.State, GridWorld.Reward> = Map [(3,2), 1.0 ; (3,1), -1.0]
let turnProbability = 0.1
let livingReward = 0.0

let mdp_x = GridWorld.getStates maximumX maximumY blockedStates
let mdp_a = GridWorld.getActions
let mdp_tf: RLCore.TransitionFunction<GridWorld.State, GridWorld.Action> =
    GridWorld.transitionfunction maximumX maximumY blockedStates absorbitionStates turnProbability
let mdp_rf: GridWorld.State->GridWorld.Action->GridWorld.State->GridWorld.Reward =
    GridWorld.rewardFunction absorbitionStates livingReward
let mdp_si = GridWorld.stateIndex maximumX maximumY
let mdp_ai = GridWorld.actionIndex
printfn "%A" mdp_x
let mdp: ValueIteration.MDP<GridWorld.State, GridWorld.Action> = {X = mdp_x ; A = mdp_a ; p = mdp_tf ; r = mdp_rf ;
si = mdp_si ; ai = mdp_ai }    

let policy (s: GridWorld.State) : GridWorld.Action = LanguagePrimitives.EnumOfValue<int, GridWorld.Action>(RLCore.rand.NextInt64(4) |> int)

let Q = ValueIteration.valueIteration mdp 1.0 

for x in mdp.X do
    for a in mdp.A do
        printfn "%s %s: %.3f" (GridWorld.stateToString x) (GridWorld.actionToString a) Q[mdp_si x, mdp_ai a]

//let env: QLearning.Env<State, Action> = {X = mdp_x ; A = mdp_a ; absorbtionStates = [absorbtionMarker] ;
//p = (convertTransitionFunction2 btf1) ; r = rf1 ; si = si ; ai = int}
//let Q2 = QLearning.qLearn env policy (1,1) 0.9
//printfn "Q-learning"
//for x in mdp.X do
//    for a in mdp.A do
//        printfn "%s %s: %.3f" (stateToString x) (actionToString a) Q2[si x, int a]
