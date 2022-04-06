module RiverSwim

open System
open QLearning
open RLCore

type State = int
type Action = 
    | Left = 0
    | Right = 1
type Reward = float

let stateToString (s: State) : string =
    $"({s |> int})"

let actionToString (a: Action) : string =
    match a with
    | Action.Left -> "Left "
    | Action.Right -> "Right"
    | _ -> ""

/// Gets all the states for a world of the specified size
/// It is simply all combinations of x and y as well as the absorptionMarker
let getStates size  =
    [0 .. size-1]

let getActions: Action list = [Action.Left; Action.Right]

let applyActionSimple action stateFrom : State =
    match action with
    | Action.Left -> stateFrom - 1
    | Action.Right -> stateFrom + 1
    | _ -> raise (ArgumentException("Error"))

let baseTransitionFunction size (reverseProbability: float)  (stayProbability: float) stateFrom action : (State * float) list =
    if (stateFrom = 0 && action = Action.Left) then 
        [0, 1]
    else if (stateFrom = 0 && action = Action.Right) then 
        [0, (stayProbability + reverseProbability); 1, (1.0 - stayProbability - reverseProbability)]
    else if (stateFrom = (size-1) && action = Action.Right) then
        [stateFrom, (stayProbability + reverseProbability) ; applyActionSimple Action.Left stateFrom, (1.0 - stayProbability - reverseProbability)]
    else if (action = Action.Left) then
        [applyActionSimple action stateFrom, 1.0]
    else
        [applyActionSimple Action.Left stateFrom, reverseProbability ; stateFrom, stayProbability ;
        applyActionSimple Action.Right stateFrom, (1.0 - reverseProbability - stayProbability)]
        

let transitionfunction size (reverseProbability: float)  (stayProbability: float) : State->Action->State->float =
    let bt = baseTransitionFunction size reverseProbability stayProbability
    RLCore.convertTransitionFunction bt

let transitionfunctionStochastic size reverseProbability  stayProbability: State->Action->State =
    let bt = baseTransitionFunction size reverseProbability stayProbability
    QLearning.convertTransitionFunctionStochastic bt

let rewardFunction (size: int) (livingReward: Reward) (rightReward: Reward) (leftReward: Reward) stateFrom action : Reward =
    if (stateFrom = 0) then
        leftReward
    else if (stateFrom = (size-1)) then
        rightReward
    else
        livingReward
        

let stateIndex size state : int = int state

let actionIndex (action: Action) : int = int action

let printV size (Q: float[,]) : unit =
    [for x in 0 .. (size-1) do getStateValue (stateIndex size) x Q |>
                                                    sprintf "+%.2f" ] |>
                                                    String.concat " " |>
                                                    fun s -> s.Replace("+-", "-") |> printfn "%s"


let stateQToString size (Q: float[,]) state: string =
    let idx = stateIndex size state
    Q[idx, *] |>
            Seq.mapi (fun i v -> i, v) |> Seq.maxBy snd |> fst |>
            LanguagePrimitives.EnumOfValue<int, Action> |>
            actionToString


let printActions size (Q: float[,]) : unit =
    [for x in 0..(size-1) do (stateQToString size Q  x)] |>
        String.concat " " |> printfn "%s" 
